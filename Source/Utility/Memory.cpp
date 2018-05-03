//
// Provide efficient memory management utilities.
//

#include "Memory.h"

namespace CSRT {
	
void* aligned_malloc(size_t required_bytes, size_t alignment) {
	void* p1; // original block
	void** p2; // aligned block
	size_t offset = alignment - 1 + sizeof(void*);
	if ((p1 = (void*)malloc(required_bytes + offset)) == nullptr) {
		return nullptr;
	}
	p2 = (void**)(((size_t)(p1)+offset) & ~(alignment - 1));
	p2[-1] = p1; 
	return p2;
}

void aligned_free(void *p) {
	free(((void**)p)[-1]);
}


// Memory Allocation Functions
void *AllocAligned(size_t size) {
	return aligned_malloc(size, CSRT_L1_CACHE_LINE_SIZE);
}

void FreeAligned(void *ptr) {
	if (!ptr) return;
	aligned_free(ptr);
}

}	// namespace CSRT