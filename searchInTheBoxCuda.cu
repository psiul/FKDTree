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
#define THREADS_PER_QUERY 16
#define QUERIES_PER_BLOCK (BLOCKSIZE/THREADS_PER_QUERY)

__global__ void CUDABFSparallel (unsigned int nPoints,  float* dimensions,  unsigned int* ids,  unsigned int* results, unsigned int* debug) {
	//__shared__ int frontier[QUERIES_PER_BLOCK][MAX_FRONTIER];
	//__shared__ int frontnext[QUERIES_PER_BLOCK][THREADS_PER_QUERY];
	//__shared__ int pfound[QUERIES_PER_BLOCK][THREADS_PER_QUERY];

	unsigned int gid = threadIdx.x/THREADS_PER_QUERY;
	unsigned int pid = blockIdx.x*QUERIES_PER_BLOCK+gid;
	unsigned int tid = threadIdx.x%THREADS_PER_QUERY;

	unsigned int* frontier = debug+nPoints*2*THREADS_PER_QUERY+pid*MAX_FRONTIER;
	unsigned int* frontnext =  debug+nPoints*THREADS_PER_QUERY+pid*THREADS_PER_QUERY;
	unsigned int* pfound = debug+pid*THREADS_PER_QUERY;

	if (pid < nPoints) {
	unsigned int found = 0;
	unsigned int sync = 4;
		
	int theDepth = floor(log2((float)nPoints));
	float minPoint[NUM_DIMENSIONS];
	float maxPoint[NUM_DIMENSIONS];
	for(int i = 0; i<NUM_DIMENSIONS; ++i) {
		minPoint[i] = dimensions[nPoints*i+pid] - RANGE;
		maxPoint[i] = dimensions[nPoints*i+pid] + RANGE;
	}
		

	Queue indecesToVisit;
	indecesToVisit.front = indecesToVisit.tail = indecesToVisit.size = 0;

	if (tid == 0)
		push_back(&indecesToVisit, 0);

	unsigned int resultIndex = nPoints + MAX_RESULT_SIZE*pid;
	
	for (int depth = 0; depth < theDepth+1; ++depth) {
		if (depth == sync) {
			sync = sync + 3;

			//synchronize in frontier and balance load
			//frontnext[gid][tid] = indecesToVisit.size; 
			frontnext[tid] = indecesToVisit.size;
			__syncthreads ();

			for (unsigned int i = 1; i < THREADS_PER_QUERY; i<<=1) {
				int add;
        	                if (tid >= i) add = frontnext[tid-i];
                        	       // add = frontnext[gid][tid-i];
                        	__syncthreads();
                        	if (tid >= i) frontnext[tid] += add;
                                	//frontnext[gid][tid] += add;
				__syncthreads();
	                }

			//unsigned int active = frontnext[gid][THREADS_PER_QUERY-1];
			//unsigned int len = indecesToVisit.size;
			//unsigned int offset = frontnext[gid][tid]-len;
			unsigned int active = frontnext[THREADS_PER_QUERY-1];
                        unsigned int len = indecesToVisit.size;
                        unsigned int offset = frontnext[tid]-len;
			//unsigned int extra = active%THREADS_PER_QUERY;
			//unsigned int extrapred = (tid < extra)?tid:extra;		

			//unsigned int r_start = (active/THREADS_PER_QUERY)*tid+extrapred;
			//unsigned int r_end = r_start+active/THREADS_PER_QUERY+(tid < extra);

			//unsigned int w_end = frontnext[gid][tid];
			//unsigned int w_start = w_end-indecesToVisit.size;
			
			//if (pid == 0) printf ("r_start= %d r_end= %d\n", r_start, r_end);
			
			for (int i = 0; i < len; i++) {
				frontier[i+offset] = pop_front(&indecesToVisit);
				//frontier[gid][i+offset] = pop_front(&indecesToVisit);
			}
			__syncthreads ();
		
			for (int i = tid; i < active; i+=THREADS_PER_QUERY) {
				push_back(&indecesToVisit, frontier[i]);
				//push_back(&indecesToVisit, frontier[gid][i]);
			}

		}		
		
		int dimension = depth % NUM_DIMENSIONS;
		unsigned int toVisit = indecesToVisit.size;

		for (unsigned int visited = 0; visited < toVisit; visited++) {
			unsigned int index = pop_front(&indecesToVisit);

			//if (pid == 0)
			//	printf ("Visited %d\n", index);

			bool intersection = intersects(index, dimensions, nPoints, minPoint, maxPoint, dimension);

			if (intersection && isInTheBox(index,dimensions,nPoints,minPoint,maxPoint)) {
				if (found < MAX_RESULT_SIZE) {
					//results[resultIndex] = index;
					resultIndex++;
					found++;
				}
			}
			
			bool isLowerThanBoxMin = dimensions[nPoints*dimension+index] < minPoint[dimension];
			int startSon = isLowerThanBoxMin;
			int endSon = startSon || intersection;

			for (int whichSon = startSon; whichSon < endSon + 1; ++whichSon) {
				unsigned int indexToAdd = leftSonIndex(index)+whichSon;
				if (indexToAdd < nPoints)
					push_back(&indecesToVisit, indexToAdd);
			}
		}			
	}
	
	//pfound[gid][tid] = found;
	pfound[tid] = found;
	__syncthreads();

	for (unsigned int s = THREADS_PER_QUERY/2; s > 0; s >>= 1) {
		if (tid < s)pfound[tid] += pfound[tid+s];
			//pfound[gid][tid] += pfound[gid][tid+s];
		__syncthreads();
	}

	if (tid == 0) {
		results[pid] = pfound[0];
		//printf ("%d\n", pfound[0]);
		//results[pid] = pfound[gid][0];
	}

	__syncthreads ();
	}
}


void CUDAKernelWrapper(unsigned int nPoints,float *d_dim,unsigned int *d_ids,unsigned int *d_results)
{


    // Number of thread blocks
    unsigned int gridSize = (int)ceil((float)nPoints/BLOCKSIZE);
    
    CUDASearchInTheKDBox<<<gridSize, BLOCKSIZE>>>(nPoints, d_dim, d_ids,d_results);


    
}

