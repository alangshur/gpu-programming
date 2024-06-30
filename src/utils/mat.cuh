#include <iostream>
#include <vector>
#include <random>
#include <type_traits>

std::vector<float>
create_mat(size_t m, size_t n, float val);

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

std::vector<float>
matmul(std::vector<float> x, std::vector<float> y, size_t m, size_t n, size_t k);

bool
is_equal(std::vector<float> x, std::vector<float> y, float tol = 1e-4);

template <typename T1, typename T2>
std::vector<T2>
cast_mat(const std::vector<T1> &vec);