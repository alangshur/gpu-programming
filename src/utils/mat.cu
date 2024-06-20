#include <iostream>
#include <vector>
#include <random>

#include "mat.cuh"

template <typename T>
std::vector<T>
create_random_mat(size_t m, size_t n, T min, T max)
{
    // seed random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min, max);

    // create matrix
    std::vector<T> mat(m * n);
    for (int i = 0; i < m * n; ++i) mat[i] = dist(gen);
    return mat;
}

template <typename T>
std::vector<T>
create_random_vec(size_t n, T min, T max)
{
    // seed random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min, max);

    // create vector
    std::vector<T> vec(n);
    for (int i = 0; i < n; ++i) vec[i] = dist(gen);
    return vec;
}

template std::vector<float>
create_random_mat<float>(size_t, size_t, float, float);
template std::vector<double>
create_random_mat<double>(size_t, size_t, double, double);

template std::vector<float>
create_random_vec<float>(size_t, float, float);
template std::vector<double>
create_random_vec<double>(size_t, double, double);