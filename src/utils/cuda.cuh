#pragma once

#include <iostream>

#define CUDA_CALL(call)                                                           \
    do                                                                            \
    {                                                                             \
        cudaError_t err = call;                                                   \
        if (cudaSuccess != err)                                                   \
        {                                                                         \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__; \
            std::cerr << ": " << cudaGetErrorString(err) << std::endl;            \
            throw std::runtime_error("CUDA call failed");                         \
        }                                                                         \
    } while (0)
