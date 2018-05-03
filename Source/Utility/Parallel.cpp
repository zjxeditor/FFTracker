//
// Provide Parallel Support.
//

#include "Parallel.h"
#include <list>
#include <thread>

namespace CSRT {

static int ThreadCount = 0;
static std::vector<std::thread> Threads;
static bool ShutdownThreads = false;
class ParallelForLoop;
static ParallelForLoop *WorkList = nullptr;
static std::mutex WorkListMutex;
static std::condition_variable WorkListCondition;

static std::function<void()> BackgroundWork;
static std::mutex BackgroundMutex;
static std::condition_variable BackgroundCondition;

thread_local int ThreadIndex = 0;

// Hold the parallel execution information for concurrency processing.
class ParallelForLoop {
public:
	// ParallelForLoop Public Methods
	ParallelForLoop(std::function<void(int64_t)> func1D, int64_t maxIndex, int chunkSize)
		: func1D(std::move(func1D)), maxIndex(maxIndex), chunkSize(chunkSize) {}
	ParallelForLoop(const std::function<void(Vector2i)> &f, const Vector2i &count)
		: func2D(f), maxIndex(count.x * count.y), chunkSize(1) {
		nX = count.x;
	}

	// ParallelForLoop Private Data
	std::function<void(int64_t)> func1D;
	std::function<void(Vector2i)> func2D;
	const int64_t maxIndex;
	const int chunkSize;
	int64_t nextIndex = 0;
	int activeWorkers = 0;
	ParallelForLoop *next = nullptr;
	int nX = -1;

	// ParallelForLoop Private Methods
	bool Finished() const {
		return nextIndex >= maxIndex && activeWorkers == 0;
	}
};

inline static void RemoveLoop(ParallelForLoop *loop) {
	if (!loop) return;
	if (WorkList == loop) {
		WorkList = loop->next;
		return;
	}
	ParallelForLoop *temp = WorkList;
	while (temp != nullptr && temp->next != loop) temp = temp->next;
	if (temp == nullptr) return;
	temp->next = loop->next;
	loop->next = nullptr;
}

// Each addtional thread's work function.
static void WorkerThreadFunc(int tIndex, std::shared_ptr<Barrier> barrier) {
	// Initialize thread.
	Info("Started execution in worker thread " + std::to_string(tIndex));
	ThreadIndex = tIndex;

	// The main thread sets up a barrier so that it can be sure that all
	// workers have finished the initialization work before it continues.
	barrier->Wait();
	// Release our reference to the Barrier so that it's freed once all of
	// the threads have cleared it.
	barrier.reset();

	std::unique_lock<std::mutex> lock(WorkListMutex);
	while (!ShutdownThreads) {
		if (!WorkList) {
			// Sleep until there are more tasks to run.
			WorkListCondition.wait(lock);
		} else {
			// Get work from _workList_ and run loop iterations.
			ParallelForLoop &loop = *WorkList;

			// Run a chunk of loop iterations for _loop_

			// Find the set of loop iterations to run next
			int64_t indexStart = loop.nextIndex;
			int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);

			// Update _loop_ to reflect iterations this thread will run
			loop.nextIndex = indexEnd;
			if (loop.nextIndex == loop.maxIndex) RemoveLoop(&loop);
			loop.activeWorkers++;

			// Run loop indices in _[indexStart, indexEnd)_
			lock.unlock();
			for (int64_t index = indexStart; index < indexEnd; ++index) {
				if (loop.func1D) {
					loop.func1D(index);
				}
				// Handle other types of loops
				else {
					if (!loop.func2D)
						Warning("Cannot find target method for concurrency processing.");
					else
						loop.func2D(Vector2i(index % loop.nX, index / loop.nX));
				}
			}
			lock.lock();

			// Update _loop_ to reflect completion of iterations
			loop.activeWorkers--;
			if (loop.Finished()) WorkListCondition.notify_all();
		}
	}

	Info("Exiting worker thread " + std::to_string(tIndex));
}

// Background thread function.
static void BackgroundFunc(int tIndex, std::shared_ptr<Barrier> barrier) {
	// Initialize thread.
	Info("Started the background thread " + std::to_string(tIndex));
	ThreadIndex = tIndex;

	// The main thread sets up a barrier so that it can be sure that all
	// workers have finished the initialization work before it continues.
	barrier->Wait();
	// Release our reference to the Barrier so that it's freed once all of
	// the threads have cleared it.
	barrier.reset();

	std::unique_lock<std::mutex> lock(BackgroundMutex);
	while (!ShutdownThreads) {
		if (!BackgroundWork) {
			BackgroundCondition.wait(lock);
		} else {
			if (!BackgroundWork)
				Warning("Cannot find any background work.");
			else
				BackgroundWork();
			BackgroundWork = nullptr;
		}
	}

	Info("Exiting background thread.");
}

void SetThreadCount(int count) {
	ThreadCount = std::max(count, 2);
}

int NumSystemCores() {
	return std::max(1u, std::thread::hardware_concurrency());
}

int MaxThreadIndex() {
	return ThreadCount == 0 ? NumSystemCores() : ThreadCount;
}

void ParallelInit() {
	if(Threads.size() != 0) Critical("Thread pool is not empty before initialzation.");
	int nThreads = MaxThreadIndex();
	if (nThreads < 2) Critical("A lease 2 threads should be required. One for foregorund task, one for background tasks.");
	ThreadIndex = 0;

	std::shared_ptr<Barrier> barrier = std::make_shared<Barrier>(nThreads);

	// Launch one fewer worker thread than the total number we want doing
	// work, since the main thread helps out, too.
	for (int i = 0; i < nThreads - 2; ++i)
		Threads.push_back(std::thread(WorkerThreadFunc, i + 1, barrier));
	Threads.push_back(std::thread(BackgroundFunc, nThreads - 1, barrier));

	barrier->Wait();
}

void ParallelCleanup() {
	if (Threads.empty()) return;

	WorkListMutex.lock();
	BackgroundMutex.lock();
	ShutdownThreads = true;
	WorkListCondition.notify_all();
	BackgroundCondition.notify_all();
	WorkListMutex.unlock();
	BackgroundMutex.unlock();

	for (std::thread &thread : Threads) thread.join();
	Threads.erase(Threads.begin(), Threads.end());
	ShutdownThreads = false;
}

void ParallelFor(std::function<void(int64_t)> func, int64_t count, int chunkSize) {
	if (Threads.size() <= 1 && MaxThreadIndex() != 2)
		Critical("Invalid thread configuration. There should be at least 2 thread (main thread and background thread).");

	// Run iterations immediately if not using threads or if _count_ is small.
	if (Threads.size() <= 1 || count < chunkSize) {
		for (int64_t i = 0; i < count; ++i) func(i);
		return;
	}

	// Create and enqueue _ParallelForLoop_ for this loop
	ParallelForLoop loop(std::move(func), count, chunkSize);
	WorkListMutex.lock();
	loop.next = WorkList;
	WorkList = &loop;
	WorkListMutex.unlock();

	// Notify worker threads of work to be done
	std::unique_lock<std::mutex> lock(WorkListMutex);
	WorkListCondition.notify_all();

	// Help out with parallel loop iterations in the current thread
	while (!loop.Finished()) {
		// Run a chunk of loop iterations for _loop_

		// Find the set of loop iterations to run next
		int64_t indexStart = loop.nextIndex;
		int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);

		// Update _loop_ to reflect iterations this thread will run
		loop.nextIndex = indexEnd;
		if (loop.nextIndex == loop.maxIndex) RemoveLoop(&loop);
		loop.activeWorkers++;

		// Run loop indices in _[indexStart, indexEnd)_
		lock.unlock();
		for (int64_t index = indexStart; index < indexEnd; ++index) {
			if (loop.func1D) {
				loop.func1D(index);
			}
			// Handle other types of loops
			else {
				if (!loop.func2D)
					Warning("Cannot find target method for concurrency processing.");
				else
					loop.func2D(Vector2i(index % loop.nX, index / loop.nX));
			}
		}
		lock.lock();

		// Update _loop_ to reflect completion of iterations
		loop.activeWorkers--;
	}
}

void ParallelFor2D(std::function<void(Vector2i)> func, const Vector2i &count) {
	if (Threads.size() <= 1 && MaxThreadIndex() != 2)
		Critical("Invalid thread configuration. There should be at least 2 thread (main thread and background thread).");

	if (Threads.size() <= 1 || count.x * count.y <= 1) {
		for (int y = 0; y < count.y; ++y)
			for (int x = 0; x < count.x; ++x) func(Vector2i(x, y));
		return;
	}

	ParallelForLoop loop(std::move(func), count);
	WorkListMutex.lock();
	loop.next = WorkList;
	WorkList = &loop;
	WorkListMutex.unlock();

	std::unique_lock<std::mutex> lock(WorkListMutex);
	WorkListCondition.notify_all();

	// Help out with parallel loop iterations in the current thread
	while (!loop.Finished()) {
		// Run a chunk of loop iterations for _loop_

		// Find the set of loop iterations to run next
		int64_t indexStart = loop.nextIndex;
		int64_t indexEnd = std::min(indexStart + loop.chunkSize, loop.maxIndex);

		// Update _loop_ to reflect iterations this thread will run
		loop.nextIndex = indexEnd;
		if (loop.nextIndex == loop.maxIndex) RemoveLoop(&loop);
		loop.activeWorkers++;

		// Run loop indices in _[indexStart, indexEnd)_
		lock.unlock();
		for (int64_t index = indexStart; index < indexEnd; ++index) {
			if (loop.func1D) {
				loop.func1D(index);
			}
			// Handle other types of loops
			else {
				if (!loop.func2D)
					Warning("Cannot find target method for concurrency processing.");
				else
					loop.func2D(Vector2i(index % loop.nX, index / loop.nX));
			}
		}
		lock.lock();

		// Update _loop_ to reflect completion of iterations
		loop.activeWorkers--;
	}
}

void BackgroundProcess(std::function<void()> func) {
	if (Threads.size() <= 0) {
		Critical("Invalid background thread configuration.");
		return;
	}

	BackgroundMutex.lock();
	BackgroundWork = func;
	BackgroundCondition.notify_all();
	BackgroundMutex.unlock();
}

void BackgroundWait() {
	BackgroundMutex.lock();
	BackgroundMutex.unlock();
}

}	// namespace CSRT