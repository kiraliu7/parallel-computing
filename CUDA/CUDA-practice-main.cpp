
// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_cuda.h>
#include <helper_timer.h>
#include <math.h> 

#include "sortingNetworks_common.h"

//this method recursively sort an array by performing bitonic sort on GPU
uint recBitonicSort(uint *h_InputVal, uint *h_OutputValGPU, uint *d_InputVal, uint *d_OutputVal, uint arrayLength, uint N, uint local_dir, cudaError_t error){
    
    uint threadCount = 0;
    
    //Base case
    //If the whole array fits into GPU memory
    if(arrayLength<=N){
        //allocate input and output array on gpu
        error = cudaMalloc((void **)&d_InputVal,  arrayLength * sizeof(uint));
        checkCudaErrors(error);
        error = cudaMalloc((void **)&d_OutputVal, arrayLength * sizeof(uint));
        checkCudaErrors(error);
        //copy the array to GPU memory
        error = cudaMemcpy(d_InputVal, h_InputVal, arrayLength * sizeof(uint), cudaMemcpyHostToDevice);
        checkCudaErrors(error);

        //sync
        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        //call bitonicsort directly on GPU
        threadCount = bitonicSort(d_OutputVal,d_InputVal,arrayLength,local_dir);
        
        //sync
        error = cudaDeviceSynchronize();
        checkCudaErrors(error);

        //copy result back to host
        error = cudaMemcpy(h_OutputValGPU, d_OutputVal, arrayLength * sizeof(uint), cudaMemcpyDeviceToHost);
        checkCudaErrors(error);

        cudaFree(d_OutputVal);
        cudaFree(d_InputVal);
    }
    //if size cannot fit on GPU in once
    else{
        //make recursive call to get a sorted first half (ascending) and sorted second half (descending), and two halves combined will be a bitonic sequence
        threadCount = recBitonicSort(h_InputVal, h_OutputValGPU, d_InputVal, d_OutputVal, arrayLength/2, N, local_dir, error);
        threadCount = recBitonicSort(h_InputVal + arrayLength/2, h_OutputValGPU + arrayLength/2, d_InputVal, d_OutputVal, arrayLength/2, N, 1-local_dir, error);
        
        //reorder to get arrayLength/N chunks so that every elment in one chunk would not be greater than every element in the next chunk
        //by comparing and potentially swaping elements with a particular distance in multiple iteration. distance/=2 after each iteration
        //after this step, the array will be sorted chunk-wise (every elment in the chunk would not be greater than every element in the next chunk), 
        //but not element wise (each chunk itself is not sorted)

        //distance between the two elements being compared
        uint stride=arrayLength/2;
        //number of chunks
        uint factor=arrayLength/N;
        uint halfN=N/2;
        uint phase=1;

        uint *d_NChunkIn, *d_NChunkOut;

        error = cudaMalloc((void **)&d_NChunkIn,  N * sizeof(uint));
        checkCudaErrors(error);
        error = cudaMalloc((void **)&d_NChunkOut,  N * sizeof(uint));
        checkCudaErrors(error);

        //when stride=halfN (chunk size=N), the chunks can now fit into GPU memory so we stop
        //stride halves after every iteration
        while(stride>halfN){
            //phase is equal to how many chunks the array is divided into that fits the description (chunk num*=2, chunk size/=2 after each iteration)
            //in first iteration, we have 1 chunk because no swapping is done yet
            //in second iteration, we will have two chunks as we swaped some elements in 1st iteration and now no element in the first chunk is greater than any element in second chunk
            for(uint j=0; j<phase; j++){
                for(uint i=0; i<factor/phase; i++){
                    //determine where the segement to give to GPU starts
                    uint start=j*(arrayLength/phase)+halfN*i;

                    //we compare 2^20 elments every time (2^10 in first half, 2^10 in second half)
                    error = cudaMemcpy(d_NChunkIn, h_OutputValGPU+start, halfN * sizeof(uint), cudaMemcpyHostToDevice);
                    checkCudaErrors(error);
                    error = cudaMemcpy(d_NChunkIn+halfN, h_OutputValGPU+(start+stride), halfN * sizeof(uint), cudaMemcpyHostToDevice);
                    checkCudaErrors(error);

                    //cuda method to do the comparison
                    bitonicCompare(d_NChunkOut, d_NChunkIn, N, local_dir);

                    //sync
                    error = cudaDeviceSynchronize();
                    checkCudaErrors(error);

                    //copy result back to host
                    error = cudaMemcpy(h_OutputValGPU+start, d_NChunkOut, halfN * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);
                    error = cudaMemcpy(h_OutputValGPU+(start+stride), d_NChunkOut+halfN, halfN * sizeof(uint), cudaMemcpyDeviceToHost);
                    checkCudaErrors(error);
                }
            }
            phase=phase*2;
            stride=stride/2;
        }

        cudaFree(d_NChunkIn);
        cudaFree(d_NChunkOut),

        //perform bitonic merge on each chunk, so that each chunk is sorted within itself
        //since the array is sorted chunk-wise already, once each chunk is sorted, the whole array would be sorted
        error = cudaMalloc((void **)&d_InputVal,  N * sizeof(uint));
        checkCudaErrors(error);

        //call bitonic sort on GPU for each chunk because they now fits on GPU
        for(uint i=0; i<factor; i++){  
            error = cudaMemcpy(d_InputVal, h_OutputValGPU+i*N, N * sizeof(uint), cudaMemcpyHostToDevice);
            checkCudaErrors(error);
            
            bitonicSort(d_InputVal, d_InputVal, N, local_dir);

            //sync
            error = cudaDeviceSynchronize();
            checkCudaErrors(error);

            //copy result back to host
            error = cudaMemcpy(h_OutputValGPU+i*N, d_InputVal, N * sizeof(uint), cudaMemcpyDeviceToHost);
            checkCudaErrors(error);
        }
        cudaFree(d_InputVal);
    }

    return threadCount;
}

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    cudaError_t error;
    printf("%s Starting...\n\n", argv[0]);

    printf("Starting up CUDA context...\n");
    int dev = findCudaDevice(argc, (const char **)argv);

    uint *h_InputVal, *h_OutputValGPU;
    uint *d_InputVal, *d_OutputVal;
    StopWatchInterface *hTimer = NULL;

    const uint         power = atoi(argv[1]);
    const uint             N = 1048576;
    const uint           DIR = 1;
    const uint     numValues = 65536;
    const uint   arrayLength = pow(2, power);

    printf("Allocating and initializing host arrays...\n\n");
    sdkCreateTimer(&hTimer);
    h_InputVal     = (uint *)malloc(arrayLength * sizeof(uint));
    h_OutputValGPU = (uint *)malloc(arrayLength * sizeof(uint));
    srand(2001);

    for (uint i = 0; i < arrayLength; i++)
    {   
        h_InputVal[i] = rand() % numValues;
    }

    printf("Testing array length %u ...\n", arrayLength);

    int flag = 1;
    printf("Running GPU bitonic sort (%u * N size), direction %u...\n\n", arrayLength/N, DIR);

    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    //recursively sort the array with bitonic sort performed on GPU
    uint threadCount = recBitonicSort(h_InputVal, h_OutputValGPU, d_InputVal, d_OutputVal, arrayLength, N, DIR, error);
    
    sdkStopTimer(&hTimer);
    printf("Time: %f ms\n\n", sdkGetTimerValue(&hTimer) );

    double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer);
    printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
        (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, arrayLength, 1, threadCount);

    printf("\nValidating the results...\n");

    //result validation
    int ValsFlag = validateSortedVals(h_OutputValGPU, h_InputVal, arrayLength, numValues, DIR);
    flag = flag && ValsFlag;

    printf("\n");

    printf("Shutting down...\n");
    printf("------------------------------------------------------------------------------------\n");
    sdkDeleteTimer(&hTimer);
    free(h_OutputValGPU);
    free(h_InputVal);

    exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
}
