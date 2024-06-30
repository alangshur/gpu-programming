#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <mma.h>

#include "mat.cuh"

std::vector<float>
create_mat(size_t m, size_t n, float val)
{
    std::vector<float> mat(m * n);
    for (int i = 0; i < m * n; ++i) mat[i] = val;
    return mat;
}

std::vector<float>
create_random_mat(size_t m, size_t n, float min, float max)
{
    // seed random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    // create matrix
    std::vector<float> mat(m * n);
    for (int i = 0; i < m * n; ++i) mat[i] = dist(gen);

    return mat;
}

std::vector<float>
create_incremental_mat(size_t m, size_t n)
{
    std::vector<float> mat(m * n);
    for (int i = 0; i < m * n; ++i) mat[i] = i;
    return mat;
}

std::vector<float>
create_random_vec(size_t n, float min, float max)
{
    // seed random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    // create vector
    std::vector<float> vec(n);
    for (int i = 0; i < n; ++i) vec[i] = dist(gen);

    return vec;
}

void
print_mat(std::vector<float> mat, size_t m, size_t n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            std::cout << mat[i * n + j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void
print_vec(std::vector<float> vec, size_t n)
{
    for (int i = 0; i < n; ++i)
    {
        std::cout << vec[i] << std::endl;
    }
}

std::vector<float>
matmul(std::vector<float> x, std::vector<float> y, size_t m, size_t n, size_t k)
{
    std::vector<float> out(m * n);

    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; j++)
        {
            float acc = 0.0;

            for (size_t ki = 0; ki < k; ++ki)
            {
                acc += x[i * k + ki] * y[ki * n + j];
            }

            out[i * n + j] = acc;
        }
    }

    return out;
}

bool
is_equal(std::vector<float> x, std::vector<float> y, float tol)
{
    if (x.size() != y.size())
        return false;

    for (size_t i = 0; i < x.size(); ++i)
    {
        if (std::abs(x[i] - y[i]) > tol)
            return false;
    }

    return true;
}

template <typename T1, typename T2>
std::vector<T2>
cast_mat(const std::vector<T1> &vec)
{
    std::vector<T1> out(vec.size());

    for (size_t i = 0; i < vec.size(); ++i)
    {
        // normalize from T1 to float
        float in;
        if constexpr (std::is_same<T1, __half>::value)
            in = __half2float(vec[i]);
        else
            in = static_cast<float>(vec[i]);

        // cast from float to T2
        if constexpr (std::is_same<T2, __half>::value)
            out[i] = __float2half(in);
        else
            out[i] = static_cast<T2>(in);
    }

    return out;
}