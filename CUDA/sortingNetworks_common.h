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



////////////////////////////////////////////////////////////////////////////////
// Shortcut definition
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;



///////////////////////////////////////////////////////////////////////////////
// Sort result validation routines
////////////////////////////////////////////////////////////////////////////////
//Sorted Vals array validation (check for integrity and proper order)
extern "C" uint validateSortedVals(
    uint *resVal,
    uint *srcVal,
    uint arrayLength,
    uint numValues,
    uint dir
);

////////////////////////////////////////////////////////////////////////////////
// CUDA sorting networks
////////////////////////////////////////////////////////////////////////////////

extern "C" uint bitonicSort(
    uint *d_DstVal,
    uint *d_SrcVal,
    uint arrayLength,
    uint dir
);

////////////////////////////////////////////////////////////////////////////////
// CUDA Compare elements from first and second half of input array
////////////////////////////////////////////////////////////////////////////////
extern "C" void bitonicCompare(
    uint *d_DstVal,
    uint *d_SrcVal,
    uint arrayLength,
    uint dir
);
