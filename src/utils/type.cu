#include <vector>
#include <mma.h>

#include "type.cuh"

std::vector<double>
vec_to_double(std::vector<float> vec)
{
    std::vector<double> out(vec.size());
    for (int i = 0; i < vec.size(); ++i) out[i] = static_cast<double>(vec[i]);
    return out;
}

std::vector<__half>
vec_to_half(std::vector<float> vec)
{
    std::vector<__half> out(vec.size());
    for (int i = 0; i < vec.size(); ++i) out[i] = __float2half(vec[i]);
    return out;
}