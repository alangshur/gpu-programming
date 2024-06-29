#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mma.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void
check(T err, const char *const func, const char *const file, int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void
checkLast(const char *const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K, typename WMMA_FRAG_LAYOUT_A,
    typename WMMA_FRAG_LAYOUT_B>
__global__ void
wmma_gemm_a_col_major_b_col_major(T1 const *A, T1 const *B, T2 *C, uint32_t m, uint32_t n, uint32_t k, uint32_t lda,
    uint32_t ldb, uint32_t ldc, bool is_A_transpose, bool is_B_transpose, float alpha, float beta)
{
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / warpSize};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_FRAG_LAYOUT_A> a_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, WMMA_FRAG_LAYOUT_B> b_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> acc_frag{};
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> c_frag{};

    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    for (uint32_t ki{0}; ki < k; ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki : warpM * WMMA_M};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? warpM * WMMA_M : ki};
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? warpN * WMMA_N : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki : warpN * WMMA_N};

        if (matrix_mma_a_row_idx < (is_A_transpose ? k : m) && matrix_mma_a_col_idx < (is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < (is_B_transpose ? n : k) && matrix_mma_b_col_idx < (is_B_transpose ? k : n))
        {
            T1 const *matrix_mma_a_mptr{A + matrix_mma_a_row_idx + matrix_mma_a_col_idx * lda};
            T1 const *matrix_mma_b_mptr{B + matrix_mma_b_row_idx + matrix_mma_b_col_idx * ldb};

            nvcuda::wmma::load_matrix_sync(a_frag, matrix_mma_a_mptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, matrix_mma_b_mptr, ldb);
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    uint32_t const matrix_mma_c_row_idx{warpM * WMMA_M};
    uint32_t const matrix_mma_c_col_idx{warpN * WMMA_N};

    if (matrix_mma_c_row_idx < m && matrix_mma_c_col_idx < n)
    {
        T2 *matrix_mma_c_mptr{C + matrix_mma_c_row_idx + matrix_mma_c_col_idx * ldc};
        nvcuda::wmma::load_matrix_sync(c_frag, matrix_mma_c_mptr, ldc, nvcuda::wmma::mem_col_major);

        for (uint32_t i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        nvcuda::wmma::store_matrix_sync(matrix_mma_c_mptr, c_frag, ldc, nvcuda::wmma::mem_col_major);
    }
}

template <typename T1, typename T2>
void
launch_wmma_mm(T1 const *A, T1 const *B, T2 *C, uint32_t m, uint32_t n, uint32_t k, bool is_A_transpose,
    bool is_B_transpose, cudaStream_t stream)
{
    // Assume there is no padding in our data.
    uint32_t const lda{is_A_transpose ? k : m};
    uint32_t const ldb{is_B_transpose ? n : k};
    uint32_t const ldc{m};
    float const alpha{1.0f};
    float const beta{0.0f};

    constexpr int WMMA_M{16};
    constexpr int WMMA_N{16};
    constexpr int WMMA_K{16};

    constexpr int WARP_SIZE{32};

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // Block size of 128x4 means we have 16 (4x4) warps,
    // each warp computes a 16x16 output tile,
    // and a block computes a 64x64 output tile.
    // Each block has 4x4 warps, totalling 4x4x32 threads.
    int const num_warps_x = 4;
    int const num_warps_y = 4;
    blockDim.x = num_warps_x * WARP_SIZE;
    blockDim.y = num_warps_y;
    // Round up.
    gridDim.x = (m + (WMMA_M * num_warps_x - 1)) / (WMMA_M * num_warps_x);
    gridDim.y = (n + WMMA_N * num_warps_y - 1) / (WMMA_N * num_warps_y);

    wmma_gemm_a_col_major_b_col_major<T1, T2, WMMA_M, WMMA_N, WMMA_K, nvcuda::wmma::col_major, nvcuda::wmma::col_major>
        <<<gridDim, blockDim, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, true, true, alpha, beta);

    CHECK_LAST_CUDA_ERROR();
}

void
fill_random_float_values(float *arr, size_t n)
{
    std::default_random_engine random_engine(0);
    std::uniform_real_distribution<float> uniform_dist(-256, 256);
    for (size_t i{0}; i < n; ++i)
    {
        arr[i] = uniform_dist(random_engine);
    }
}

void
float2half(__half *half_arr, float const *float_arr, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        half_arr[i] = __float2half(float_arr[i]);
    }
}

int
main()
{
    uint32_t const matrix_size_m{1024};
    uint32_t const matrix_size_n{1024};
    uint32_t const matrix_size_k{1024};

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    std::vector<float> matrix_a_float(matrix_size_m * matrix_size_k);
    std::vector<float> matrix_b_float(matrix_size_k * matrix_size_n);
    std::vector<__half> matrix_a_half(matrix_size_m * matrix_size_k);
    std::vector<__half> matrix_b_half(matrix_size_k * matrix_size_n);
    std::vector<float> matrix_c_float(matrix_size_m * matrix_size_n);
    std::vector<float> matrix_c_float_reference(matrix_size_m * matrix_size_n);

    float *h_matrix_a_float{matrix_a_float.data()};
    float *h_matrix_b_float{matrix_b_float.data()};
    __half *h_matrix_a_half{matrix_a_half.data()};
    __half *h_matrix_b_half{matrix_b_half.data()};
    float *h_matrix_c_float{matrix_c_float.data()};
    float *h_matrix_c_float_reference{matrix_c_float_reference.data()};

    fill_random_float_values(h_matrix_a_float, matrix_a_float.size());
    fill_random_float_values(h_matrix_b_float, matrix_b_float.size());
    fill_random_float_values(h_matrix_c_float, matrix_c_float.size());
    fill_random_float_values(h_matrix_c_float_reference, matrix_c_float_reference.size());
    float2half(h_matrix_a_half, h_matrix_a_float, matrix_a_float.size());
    float2half(h_matrix_b_half, h_matrix_b_float, matrix_b_float.size());

    __half *d_matrix_a_half, *d_matrix_b_half;
    float *d_matrix_c_float;

    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_a_half, matrix_size_m * matrix_size_k * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_b_half, matrix_size_k * matrix_size_n * sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix_c_float, matrix_size_m * matrix_size_n * sizeof(float)));

    // Copy data from host to device.
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_matrix_a_half, h_matrix_a_half, matrix_a_float.size() * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(
        cudaMemcpy(d_matrix_b_half, h_matrix_b_half, matrix_b_float.size() * sizeof(__half), cudaMemcpyHostToDevice));

    // Compute matrix multiplication reference output using CUDA WMMA.
    launch_wmma_mm<half, float>(d_matrix_a_half, d_matrix_b_half, d_matrix_c_float, matrix_size_m, matrix_size_n,
        matrix_size_k, false, false, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(
        cudaMemcpy(h_matrix_c_float, d_matrix_c_float, matrix_c_float.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_matrix_a_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_b_half));
    CHECK_CUDA_ERROR(cudaFree(d_matrix_c_float));

    return 0;
}