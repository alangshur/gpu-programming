#include <iostream>
#include <vector>
#include <random>

#include "mat.cuh"

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