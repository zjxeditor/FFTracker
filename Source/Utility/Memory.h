//
// Provide efficient memory management utilities.
//

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef CSRT_MEMORY_H
#define CSRT_MEMORY_H

#include "../CSRT.h"
#include <list>
#include <cstddef>

namespace CSRT {

#define ARENA_ALLOC(arena, Type) new ((arena).Alloc(sizeof(Type))) Type
void *AllocAligned(size_t size);
template <typename T>
T *AllocAligned(size_t count) {
	return (T *)AllocAligned(count * sizeof(T));
}
void FreeAligned(void *);

// Memory Arena for dynamic memory management. Note, it is not thread safe.
// It does not support freeing of individual blocks of memory, only freeing 
// of all of the memory in the arena at once.Thus, it is useful when a umber 
// of allocations need to be done quickly and all of the allocated objects 
// have similar lifetimes.
class MemoryArena {
public:
	// MemoryArena Public Methods
	MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) {}

	~MemoryArena() {
		FreeAligned(currentBlock);
		for (auto &block : usedBlocks) FreeAligned(block.second);
		for (auto &block : availableBlocks) FreeAligned(block.second);
	}
	void *Alloc(size_t nBytes) {
		// Round up _nBytes_ to minimum machine alignment
		/*const int align = alignof(std::max_align_t);
		if (!IsPowerOf2(align))
			Critical("System minimum alignment not a power of two. Cannot use the MemoryArena class.");
		nBytes = (nBytes + align - 1) & ~(align - 1);*/
		const size_t offset = CSRT_L1_CACHE_LINE_SIZE - 1 + sizeof(void*);
		const size_t newBytes = nBytes + offset;
		void *ret = nullptr;
		
		arenaMutex.lock();
		if (currentBlockPos + newBytes > currentAllocSize) {
			// Add current block to _usedBlocks_ list
			if (currentBlock) {
				usedBlocks.push_back(
					std::make_pair(currentAllocSize, currentBlock));
				currentBlock = nullptr;
				currentAllocSize = 0;
			}

			// Get new block of memory for _MemoryArena_

			// Try to get memory block from _availableBlocks_
			for (auto iter = availableBlocks.begin(); iter != availableBlocks.end(); ++iter) {
				if (iter->first >= nBytes) {
					currentAllocSize = iter->first;
					currentBlock = iter->second;
					availableBlocks.erase(iter);
					break;
				}
			}
			if (!currentBlock) {
				currentAllocSize = std::max(nBytes, blockSize);
				currentBlock = AllocAligned<uint8_t>(currentAllocSize);
			}
			ret = currentBlock;
			currentBlockPos = nBytes;
		} else {
			void *p1 = currentBlock + currentBlockPos;
			ret = (void**)(((size_t)(p1)+offset) & ~(CSRT_L1_CACHE_LINE_SIZE - 1));
			currentBlockPos += newBytes;
		}
		arenaMutex.unlock();
		return ret;
	}

	template <typename T>
	T *Alloc(size_t n = 1, bool runConstructor = true) {
		T *ret = (T *)Alloc(n * sizeof(T));
		if (runConstructor)
			for (size_t i = 0; i < n; ++i) new (&ret[i]) T();
		return ret;
	}

	void *Realloc(void *previous, size_t oldSize, size_t newSize) {
		void *ret = Alloc(newSize);
		memcpy(ret, previous, oldSize);
		return ret;
	}

	void Reset() {
		arenaMutex.lock();
		currentBlockPos = 0;
		availableBlocks.splice(availableBlocks.begin(), usedBlocks);
		arenaMutex.unlock();
	}

	size_t TotalAllocated() const {
		size_t total = currentAllocSize;
		for (const auto &alloc : usedBlocks) total += alloc.first;
		for (const auto &alloc : availableBlocks) total += alloc.first;
		return total;
	}

private:
	MemoryArena(const MemoryArena &) = delete;
	MemoryArena &operator=(const MemoryArena &) = delete;
	// MemoryArena Private Data
	const size_t blockSize;
	size_t currentBlockPos = 0, currentAllocSize = 0;
	uint8_t *currentBlock = nullptr;
	std::list<std::pair<size_t, uint8_t *>> usedBlocks, availableBlocks;
	std::mutex arenaMutex;
};

}	// namespace CSRT

#endif	// CSRT_MEMORY_H
