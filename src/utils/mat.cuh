#include <iostream>
#include <vector>
#include <random>

std::vector<float>
create_random_mat(size_t m, size_t n, float min = 0.0f, float max = 1.0f);

std::vector<float>
create_incremental_mat(size_t m, size_t n);

std::vector<float>
create_random_vec(size_t n, float min = 0.0f, float max = 1.0f);

void
print_mat(std::vector<float> mat, size_t m, size_t n);

void
print_vec(std::vector<float> vec, size_t n);