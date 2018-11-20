//
// Provide Parallel Support.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_PARALLEL_H
#define CSRT_PARALLEL_H

#include "../CSRT.h"
#include "Geometry.h"
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace CSRT {

// Atomic float value support.
class AtomicFloat {
public:
	// AtomicFloat Public Methods
	explicit AtomicFloat(float v = 0) { bits = FloatToBits(v); }
	operator float() const { return BitsToFloat(bits); }
	float operator=(float v) {
		bits = FloatToBits(v);
		return v;
	}
	void Add(float v) {
		uint32_t oldBits = bits, newBits;
		do {
			newBits = FloatToBits(BitsToFloat(oldBits) + v);
		} while (!bits.compare_exchange_weak(oldBits, newBits));
	}

private:
	// AtomicFloat Private Data
	std::atomic<uint32_t> bits;
};

// Simple one-use barrier; ensures that multiple threads all reach a
// particular point of execution before allowing any of them to proceed
// past it.
// Note: this should be heap allocated and managed with a shared_ptr, where
// all threads that use it are passed the shared_ptr. This ensures that
// memory for the Barrier won't be freed until all threads have
// successfully cleared it.
class Barrier {
public:
	Barrier(int count) : count(count) {
		if (count <= 0)
			Critical("Invalid barrier count number.");
	}
	~Barrier() {
		if (count != 0)
			Critical("Barrier count number is not 0 when destructing.");
	}
	void Wait() {
		std::unique_lock<std::mutex> lock(mutex);
		if (count <= 0)
			Critical("Invalid barrier count number.");
		if (--count == 0)
			// This is the last thread to reach the barrier; wake up all of the
			// other ones before exiting.
			cv.notify_all();
		else
			// Otherwise there are still threads that haven't reached it. Give
			// up the lock and wait to be notified.
			cv.wait(lock, [this] { return count == 0; });
	}

private:
	std::mutex mutex;
	std::condition_variable cv;
	int count;
};

extern CSRT_API thread_local int ThreadIndex;

// Set the total thread count. Note, for computation intensive tasks, the best choice is
// the CPU core number. However, we can have each CPU core own multi threads for IO intensive tasks.
CSRT_API void SetThreadCount(int count);

CSRT_API int MaxThreadIndex();
CSRT_API int NumSystemCores();

// Call "ParallelInit" method and "ParallelCleanup" once just at application start and end.
CSRT_API void ParallelInit();
CSRT_API void ParallelCleanup();

// Concurrency processing. Parallel call methods can be safely nested.
CSRT_API void ParallelFor(std::function<void(int64_t)> func, int64_t count, int chunkSize = 1);
CSRT_API void ParallelFor2D(std::function<void(Vector2i)> func, const Vector2i &count);

// Background processing.
CSRT_API void BackgroundProcess(std::function<void()> func);
CSRT_API void BackgroundWait();

}	// namespace CSRT

#endif	// CSRT_PARALLEL_H
