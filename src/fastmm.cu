#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <mma.h>

#include "utils/cuda.cuh"
#include "fastmm.cuh"

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

template <typename WMMA_T1, typename WMMA_T2>
__global__ void
wmma_mm(size_t m, size_t n, size_t k, WMMA_T1 *x, WMMA_T1 *y, WMMA_T2 *out)
{
    // find indices of the current warp's WMMA_MxWMMA_N output tile
    size_t warp_m = blockIdx.y * blockDim.y + threadIdx.y;
    size_t warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    // initialize the WMMA fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, WMMA_T1, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, WMMA_T1, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, WMMA_T2> acc_frag;

    // make sure accumulator starts from zero
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<WMMA_T2>(0));

    // loop over k/WMMA_K tiles to calculate the output tile
    for (size_t ki = 0; ki < k; ki += WMMA_K)
    {
        // find the row and column of the A tile from X
        size_t wmma_x_row = warp_m * WMMA_M;
        size_t wmma_x_col = ki;

        // find the row and column of the B tile from Y
        size_t wmma_y_row = ki;
        size_t wmma_y_col = warp_n * WMMA_N;

        // get pointers to the current A and B tiles
        WMMA_T1 *x_ptr = x + wmma_x_row * k + wmma_x_col;
        WMMA_T1 *y_ptr = y + wmma_y_row * n + wmma_y_col;

        // load data into fragments
        nvcuda::wmma::load_matrix_sync(a_frag, x_ptr, k);
        nvcuda::wmma::load_matrix_sync(b_frag, y_ptr, n);

        // perform matrix multiplication
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    // find the row and column of the output tile
    size_t wmma_out_row = warp_m * WMMA_M;
    size_t wmma_out_col = warp_n * WMMA_N;

    // get pointer to the current output tile
    WMMA_T2 *out_ptr = out + wmma_out_row * n + wmma_out_col;

    // store the result in the output tile
    nvcuda::wmma::store_matrix_sync(out_ptr, acc_frag, n, nvcuda::wmma::mem_row_major);
}

template <typename T1, typename T2>
FastMM<T1, T2>::FastMM(
    const std::vector<T1> &x, const std::vector<T1> &y, const size_t m, const size_t n, const size_t k)
    : m_(m), n_(n), k_(k), x_(x), y_(y)
{
    // verify vector sizes
    if (x.size() != m_ * k_ || y.size() != k_ * n_)
        throw std::runtime_error("Vector sizes do not match");
    else if (x.size() == 0 || y.size() == 0)
        throw std::runtime_error("Vector size cannot be zero");

    // create stream
    CUDA_CALL(cudaStreamCreate(&stream_));

    // create events
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    // allocate device memory
    CUDA_CALL(cudaMallocAsync(&d_x, m_ * k_ * sizeof(T1), stream_));
    CUDA_CALL(cudaMallocAsync(&d_y, k_ * n_ * sizeof(T1), stream_));
    CUDA_CALL(cudaMallocAsync(&d_out, m_ * n_ * sizeof(T2), stream_));

    // copy data to device
    CUDA_CALL(cudaMemcpyAsync(d_x, x.data(), m_ * k_ * sizeof(T1), cudaMemcpyHostToDevice, stream_));
    CUDA_CALL(cudaMemcpyAsync(d_y, y.data(), k_ * n_ * sizeof(T1), cudaMemcpyHostToDevice, stream_));
}

template <typename T1, typename T2>
FastMM<T1, T2>::~FastMM()
{
    cudaStreamSynchronize(stream_);

    // destroy stream
    cudaStreamDestroy(stream_);

    // free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
}

template <typename T1, typename T2>
void
FastMM<T1, T2>::run()
{
    const size_t num_warps_x = 4;
    const size_t num_warps_y = 4;

    dim3 threads_per_block(num_warps_x * WARP_SIZE, num_warps_y);
    dim3 blocks_per_grid((n_ + WMMA_N * num_warps_x - 1) / (WMMA_N * num_warps_x),
        (m_ + WMMA_M * num_warps_y - 1) / (WMMA_M * num_warps_y));

    // record start event
    CUDA_CALL(cudaEventRecord(start, stream_));

    // launch kernel
    wmma_mm<T1, T2><<<blocks_per_grid, threads_per_block, 0, stream_>>>(m_, n_, k_, d_x, d_y, d_out);
    CUDA_CALL(cudaGetLastError());

    // record stop event
    CUDA_CALL(cudaEventRecord(stop, stream_));
}

template <typename T1, typename T2>
std::vector<T2>
FastMM<T1, T2>::get()
{
    std::vector<T2> out(m_ * n_);
    CUDA_CALL(cudaMemcpyAsync(out.data(), d_out, m_ * n_ * sizeof(T2), cudaMemcpyDeviceToHost, stream_));
    CUDA_CALL(cudaStreamSynchronize(stream_));
    return out;
}

template <typename T1, typename T2>
float
FastMM<T1, T2>::time()
{
    float ms;
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

template class FastMM<__half, float>;
