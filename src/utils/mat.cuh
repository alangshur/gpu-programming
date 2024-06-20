#include <iostream>
#include <vector>
#include <random>

template <typename T>
std::vector<T>
create_random_mat(size_t m, size_t n, T min, T max);

template <typename T>
std::vector<T>
create_random_vec(size_t n, T min, T max);