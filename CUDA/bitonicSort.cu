/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



//Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm



#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"



////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////

//I modified this method so now it works for arrayLength<1024U
__global__ void bitonicSortShared(
    uint *d_DstVal,
    uint *d_SrcVal,
    uint arrayLength,
    uint dir
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Shared memory storage for one or more short vectors
    __shared__ uint s_Val[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subbatch and load data
    //I changed offset from SHARED_SIZE_LIMIT to arrayLength to handle cases where arrayLength<1024U
    d_SrcVal += blockIdx.x * arrayLength + threadIdx.x;
    d_DstVal += blockIdx.x * arrayLength + threadIdx.x;
    s_Val[threadIdx.x +                 0] = d_SrcVal[                 0];
    s_Val[threadIdx.x + (arrayLength / 2)] = d_SrcVal[(arrayLength / 2)];

    for (uint size = 2; size < arrayLength; size <<= 1)
    {
        //Bitonic merge
        uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_Val[pos +      0], 
                s_Val[pos + stride], 
                ddd
            );
        }
    }

    //ddd == dir for the last bitonic merge step
    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_Val[pos +      0], 
                s_Val[pos + stride], 
                dir
            );
        }
    }

    cg::sync(cta);
    d_DstVal[                0] = s_Val[threadIdx.x +                0];
    d_DstVal[(arrayLength / 2)] = s_Val[threadIdx.x + (arrayLength / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
//Bottom-level bitonic sort
//Almost the same as bitonicSortShared with the exception of
//even / odd subarrays being sorted in opposite directions
//Bitonic merge accepts both
//Ascending | descending or descending | ascending sorted pairs
__global__ void bitonicSortShared1(
    uint *d_DstVal,
    uint *d_SrcVal
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Shared memory storage for current subarray
    __shared__ uint s_Val[SHARED_SIZE_LIMIT];

    //Offset to the beginning of subarray and load data
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_Val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_Val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1)
    {
        //Bitonic merge
        uint ddd = (threadIdx.x & (size / 2)) != 0;

        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_Val[pos +      0], 
                s_Val[pos + stride], 
                ddd
            );
        }
    }

    //Odd / even arrays of SHARED_SIZE_LIMIT elements
    //sorted in opposite directions
    uint ddd = blockIdx.x & 1;
    {
        for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(
                s_Val[pos +      0], 
                s_Val[pos + stride], 
                ddd
            );
        }
    }


    cg::sync(cta);
    d_DstVal[                      0] = s_Val[threadIdx.x +                       0];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_Val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

//Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal(
    uint *d_DstVal,
    uint *d_SrcVal,
    uint arrayLength,
    uint size,
    uint stride,
    uint dir
)
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint        comparatorI = global_comparatorI & (arrayLength / 2 - 1);

    //Bitonic merge
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    uint ValA = d_SrcVal[pos +      0];
    uint ValB = d_SrcVal[pos + stride];

    Comparator(
        ValA, 
        ValB, 
        ddd
    );

    d_DstVal[pos +      0] = ValA;
    d_DstVal[pos + stride] = ValB;
}

//Combined bitonic merge steps for
//size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(
    uint *d_DstVal,
    uint *d_SrcVal,
    uint arrayLength,
    uint size,
    uint dir
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Shared memory storage for current subarray
    __shared__ uint s_Val[SHARED_SIZE_LIMIT];

    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_Val[threadIdx.x +                       0] = d_SrcVal[                      0];
    s_Val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    //Bitonic merge
    uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1)
    {
        cg::sync(cta);
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(
            s_Val[pos +      0], 
            s_Val[pos + stride], 
            ddd
        );
    }

    cg::sync(cta);
    d_DstVal[                      0] = s_Val[threadIdx.x +                       0];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_Val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}



////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L)
{
    if (!L)
    {
        *log2L = 0;
        return 0;
    }
    else
    {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++);

        return L;
    }
}

extern "C" uint bitonicSort(
    uint *d_DstVal,
    uint *d_SrcVal,
    uint arrayLength,
    uint dir
)
{
    //Nothing to sort
    if (arrayLength < 2)
        return 0;

    //Only power-of-two array lengths are supported by this implementation
    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint blockCount, threadCount;

    //I enabled the method to handle arrayLength<1024U
    if (arrayLength <= SHARED_SIZE_LIMIT)
    {
        //for such a small size, we only need 1 block of arrayLength/2 threads (thats the number of comparisons in parallel we need)
        threadCount=arrayLength/2;
        bitonicSortShared<<<1, threadCount>>>(d_DstVal, d_SrcVal, arrayLength, dir);
    }
    else
    {
        blockCount = arrayLength / SHARED_SIZE_LIMIT;
        threadCount = SHARED_SIZE_LIMIT / 2;
        bitonicSortShared1<<<blockCount, threadCount>>>(d_DstVal, d_SrcVal);

        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
                if (stride >= SHARED_SIZE_LIMIT)
                {
                    bitonicMergeGlobal<<<blockCount, threadCount>>>(d_DstVal, d_DstVal, arrayLength, size, stride, dir);
                }
                else
                {
                    bitonicMergeShared<<<blockCount, threadCount>>>(d_DstVal, d_DstVal, arrayLength, size, dir);
                    break;
                }
    }

    return threadCount;
}


//call bitonicMergeGlobal for an array of size 2^20 (GPU limit)
extern "C" void bitonicCompare(
    uint *d_DstVal,
    uint *d_SrcVal,
    uint arrayLength,
    uint dir
)
{   
    //arrayLength will always be 2^20 when called from host code
    //Stride is always 2^10 because first half and second half are picked from different location when bitonicCompare() is called on host
    bitonicMergeGlobal<<<1024, 512>>>(d_DstVal, d_SrcVal, arrayLength, arrayLength, arrayLength/2 ,dir);
}
