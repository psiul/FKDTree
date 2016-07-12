#define MAX_SIZE 128
#define NUM_DIMENSIONS 3
#define MAX_RESULT_SIZE 512
#define RANGE 0.2f;
#define BLOCKSIZE 256
#include "cuda.h"
#include <stdlib.h>

typedef struct
{
    
    unsigned int data[MAX_SIZE];
    unsigned int front;
    unsigned int tail;
    unsigned int size;
} Queue;

__device__ bool push_back(Queue* queue, unsigned int index)
{
    if (queue->size < MAX_SIZE)
    {
        queue->data[queue->tail] = index;
        queue->tail = (queue->tail + 1) % MAX_SIZE;
        queue->size++;
        return true;
    }
    return false;
    
}

__device__ unsigned int pop_front(Queue* queue)
{
    if (queue->size > 0)
    {
        unsigned int element = queue->data[queue->front];
        queue->front = (queue->front + 1) % MAX_SIZE;
        queue->size--;
        return element;
    }
}

__device__ void erase_first_n_elements(Queue* queue, unsigned int n)
{
    unsigned int elementsToErase = queue->size - n > 0 ? n : queue->size;
    queue->size -=elementsToErase;
    queue->front = (queue->front + elementsToErase) % MAX_SIZE;
    
}


__device__ unsigned int leftSonIndex(unsigned int index)
{
    return 2 * index + 1;
}


__device__ unsigned int rightSonIndex(unsigned int index)
{
    return 2 * index + 2;
}


__device__ bool intersects(unsigned int index,  float* theDimensions, unsigned int nPoints,
                           float* minPoint, float* maxPoint, int dimension)
{
    return (theDimensions[nPoints * dimension + index] <= maxPoint[dimension]
            && theDimensions[nPoints * dimension + index] >= minPoint[dimension]);
}


__device__ bool isInTheBox(unsigned int index,  float* theDimensions, unsigned int nPoints,
                           float* minPoint, float* maxPoint)
{
    bool inTheBox = true;
    for (int i = 0; i < NUM_DIMENSIONS; ++i)
    {
        inTheBox &= (theDimensions[nPoints * i + index] <= maxPoint[i]
                     && theDimensions[nPoints * i + index] >= minPoint[i]);
    }
    
    return inTheBox;
}


__global__ void CUDASearchInTheKDBox(unsigned int nPoints,  float* dimensions,  unsigned int* ids,  unsigned int* results)
{
    
    // Global Thread ID
    unsigned int point_index = blockIdx.x*blockDim.x+threadIdx.x;
    
    //	float range = 0.1f;
    if(point_index < nPoints)
    {
        
        int theDepth = floor(log2((float)nPoints));
        float minPoint[NUM_DIMENSIONS];
        float maxPoint[NUM_DIMENSIONS];
        for(int i = 0; i<NUM_DIMENSIONS; ++i)
        {
            minPoint[i] = dimensions[nPoints*i+point_index] - RANGE;
            maxPoint[i] = dimensions[nPoints*i+point_index] + RANGE;
        }
        
        Queue indecesToVisit;
        indecesToVisit.front = indecesToVisit.tail =indecesToVisit.size =0;
        unsigned int pointsFound=0;
        unsigned int resultIndex = nPoints + MAX_RESULT_SIZE*point_index;
        push_back(&indecesToVisit, 0);
        
        for (int depth = 0; depth < theDepth + 1; ++depth)
        {
            int dimension = depth % NUM_DIMENSIONS;
            unsigned int numberOfIndecesToVisitThisDepth =
            indecesToVisit.size;
            
            for (unsigned int visitedIndecesThisDepth = 0;
                 visitedIndecesThisDepth < numberOfIndecesToVisitThisDepth;
                 visitedIndecesThisDepth++)
            {
                
                //				unsigned int index = indecesToVisit.data[(indecesToVisit.front+visitedIndecesThisDepth)% MAX_SIZE];
                unsigned int index = pop_front(&indecesToVisit);
                //				if(point_index == 0)
                //				{
                //					printf("index: %d, dimensions: %f %f %f\n", index, dimensions[index], dimensions[nPoints+index], dimensions[2*nPoints+index]);
                //				}
                
                bool intersection = intersects(index,dimensions, nPoints, minPoint, maxPoint,
                                               dimension);
                
                if(intersection && isInTheBox(index, dimensions, nPoints, minPoint, maxPoint))
                {
                    if(pointsFound < MAX_RESULT_SIZE)
                    {
                        //						if(point_index == 0)
                        //						{
                        //							printf("index: %d added to results", index);
                        //						}
                        
                        results[resultIndex] = index;
                        resultIndex++;
                        pointsFound++;
                    }
                    
                }
                
                bool isLowerThanBoxMin = dimensions[nPoints*dimension + index]
                < minPoint[dimension];
                int startSon = isLowerThanBoxMin; //left son = 0, right son =1
                
                int endSon = isLowerThanBoxMin || intersection;
                
                for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon)
                {
                    unsigned int indexToAdd = leftSonIndex(index) + whichSon;
                    
                    if (indexToAdd < nPoints)
                    {
                        push_back(&indecesToVisit,indexToAdd);
                    
                    }
                }
            }
            
            //			erase_first_n_elements(&indecesToVisit,numberOfIndecesToVisitThisDepth );
        }
        
        results[point_index] = pointsFound;
        
    }
    
}

#define MAX_FRONTIER 256

__global__ void CUDABFSparallel (unsigned int nPoints,  float* dimensions,  unsigned int* ids,  unsigned int* results) {
	__shared__ int frontier[MAX_FRONTIER];
	__shared__ int frontnext[MAX_FRONTIER];
	__shared__ int pfound[MAX_FRONTIER];

	unsigned int point_index = blockIdx.x;
	unsigned int thread_index = threadIdx.x;

	unsigned int found = 0;

	if (point_index < nPoints) {
		int theDepth = floor(log2((float)nPoints));
		float minPoint[NUM_DIMENSIONS];
		float maxPoint[NUM_DIMENSIONS];
        for(int i = 0; i<NUM_DIMENSIONS; ++i) {
			minPoint[i] = dimensions[nPoints*i+point_index] - RANGE;
			maxPoint[i] = dimensions[nPoints*i+point_index] + RANGE;
		}

		//if (thread_index < MAX_FRONTIER) {
			frontier[0] = 0;

			int startSon;
			int endSon = startSon-1;

			for (int depth = 0; depth < theDepth+1; ++depth) {
				int dimension = depth % NUM_DIMENSIONS;

				unsigned int index = frontier[thread_index];
				frontier[thread_index] = nPoints;

				if (index < nPoints) {
					int intersection = intersects(index, dimensions,nPoints,minPoint,maxPoint,dimension);

					if (intersection && isInTheBox(index,dimensions,nPoints,minPoint,maxPoint)) {
						if (found < MAX_RESULT_SIZE) {
								//results[resultIndex] = index;
								resultIndex++;
								found++;
						}
					}

					startSon = dimensions[nPoints*dimension+index] < minPoint[dimension];
					endSon = startSon || intersection;
				}

				int next = 0;

				for (int whichSon = startSon; whichSon < endSon+1; ++whichSon)
					next += (leftSonIndex(index) + whichSon < nPoints);
				frontnext[thread_index] = next;

				__syncthreads();

				thrust::exclusive_scan(frontnext,frontnext+MAX_FRONTIER,frontnext);				
				write_offset = frontnext[thread_index];

				for (int i = startSon; i < startSon+next; i++, write_offset++) {
					frontier[write_offset] = leftSonIndex(index)+i;
				}

				__syncthreads();

				endSon = startSon-1;
			}

			pfound[thread_index] = found;

			unsigned int pointsFound = thrust::reduce(pfound, pfound+MAX_FRONTIER);

			results[point_index] = pointsFound;
		//}
	}
}

void CUDAKernelWrapper(unsigned int nPoints,float *d_dim,unsigned int *d_ids,unsigned int *d_results)
{


    // Number of thread blocks
    unsigned int gridSize = (int)ceil((float)nPoints/BLOCKSIZE);
    
    CUDASearchInTheKDBox<<<gridSize, BLOCKSIZE>>>(nPoints, d_dim, d_ids,d_results);


    
}

