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


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sortingNetworks_common.h"



////////////////////////////////////////////////////////////////////////////////
// Validate sorted Vals array (check for integrity and proper order)
////////////////////////////////////////////////////////////////////////////////
extern "C" uint validateSortedVals(
    uint *resVal,
    uint *srcVal,
    uint arrayLength,
    uint numValues,
    uint dir
)
{
    uint *srcHist;
    uint *resHist;

    if (arrayLength < 2)
    {
        printf("validateSortedVals(): arrayLength too short, single element array is always sorted\n");
        return 1;
    }

    printf("...inspecting Vals array: ");

    srcHist = (uint *)malloc(numValues * sizeof(uint));
    resHist = (uint *)malloc(numValues * sizeof(uint));

    int flag = 1;

    //Build histograms for Vals arrays
    memset(srcHist, 0, numValues * sizeof(uint));
    memset(resHist, 0, numValues * sizeof(uint));

    for (uint i = 0; i < arrayLength; i++)
    {
        if (srcVal[i] < numValues && resVal[i] < numValues)
        {
            srcHist[srcVal[i]]++;
            resHist[resVal[i]]++;
        }
        else
        {
            flag = 0;
            break;
        }
    }

    if (!flag)
    {
        printf("***Set %u source/result Val arrays are not limited properly***\n");
        goto brk;
    }

    //Compare the histograms
    for (uint i = 0; i < numValues; i++)
        if (srcHist[i] != resHist[i])
        {
            flag = 0;
            break;
        }

    if (!flag)
    {
        printf("***Set %u source/result Vals histograms do not match***\n");
        goto brk;
    }

    if (dir)
    {
        //Ascending order
        for (uint i = 0; i < arrayLength - 1; i++)
            if (resVal[i + 1] < resVal[i])
            {
                flag = 0;
                break;
            }
    }
    else
    {
        //Descending order
        for (uint i = 0; i < arrayLength - 1; i++)
            if (resVal[i + 1] > resVal[i])
            {
                flag = 0;
                break;
            }
    }

    if (!flag)
    {
        printf("***Result Val array is not ordered properly***\n");
        goto brk;
    }

brk:
    free(resHist);
    free(srcHist);

    if (flag) printf("OK\n");

    return flag;
}
