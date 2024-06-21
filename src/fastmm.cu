#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <mma.h>

#include "utils/cuda.cuh"
#include "fastmm.cuh"

#define BLOCK_DIM 32

/*
each warp computes a 16x16x16 matrix multiplication -> resulting in a 16x16 tile of the output matrix
each block contains 4x4 warps -> resulting in a 64x64 tile of the output matrix

the thread block is arranged as a 128x4 grid of warps

> say we are performing 1024x1024x1024 matrix multiplication
each thread block will compute a 128x128 tile of the output matrix
this means our grid will be 8x8 threadblocks

if you increase the warp-wide matrix dimensions or the number of warps per block, you will use fewer threadblocks in
total
this might be inefficient if we have fewer threadblocks than the number of SMs on the GPU (wave quantization)
similarly, it might be interesting to study the trade-off between the number of warps and the warp-wide matrix
dimensions
*/

template <typename T1, typename T2, typename WMMA_M, typename WMMA_N, typename WMMA_K>
__global__ void
wmma_mm(size_t m, size_t n, size_t k, T1 *x, T1 *y, T1 *out)
{
    // these are the indices of this thread's warp
    size_t warp_m = blockIdx.y * blockDim.y + threadIdx.y;
    size_t warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    size_t warp_row = warp_m * WMMA_M;
    size_t warp_col = warp_n * WMMA_N;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> acc_frag;

    // make sure accumulator starts from zero
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    for (size_t ki = 0; ki < k; ki += WMMA_K)
    {
        size_t mma_x_row = warp_row;
        size_t mma_x_col = ki;

        size_t mma_y_row = ki;
        size_t mma_y_col = warp_col;

        if (mma_x_row < m && mma_x_col < k && mma_y_row < k && mma_y_col < n)
        {
            nvcuda::wmma::load_matrix_sync(a_frag, x + mma_x_row * k + mma_x_col, k);
            nvcuda::wmma::load_matrix_sync(b_frag, y + mma_y_row * n + mma_y_col, n);
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        // load data into fragments
        nvcuda::wmma::load_matrix_sync(a_frag, x + blockIdx.y * WMMA_M * k + frag_index * WMMA_K, k);
        nvcuda::wmma::load_matrix_sync(b_frag, y + frag_index * WMMA_K * n + blockIdx.x * WMMA_K, n);

        // perform matrix multiplication
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
}

template <typename T>
FastMM<T>::FastMM(const std::vector<T> &x, const std::vector<T> &y, const size_t m, const size_t n, const size_t k)
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
    CUDA_CALL(cudaMallocAsync(&d_x, m_ * k_ * sizeof(T), stream_));
    CUDA_CALL(cudaMallocAsync(&d_y, k_ * n_ * sizeof(T), stream_));
    CUDA_CALL(cudaMallocAsync(&d_out, m_ * n_ * sizeof(T), stream_));

    // copy data to device
    CUDA_CALL(cudaMemcpyAsync(d_x, x.data(), m_ * k_ * sizeof(T), cudaMemcpyHostToDevice, stream_));
    CUDA_CALL(cudaMemcpyAsync(d_y, y.data(), k_ * n_ * sizeof(T), cudaMemcpyHostToDevice, stream_));
}

template <typename T>
FastMM<T>::~FastMM()
{
    cudaStreamSynchronize(stream_);

    // destroy stream
    cudaStreamDestroy(stream_);

    // free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
}

template <typename T>
void
FastMM<T>::run()
{
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid((n_ + BLOCK_DIM - 1) / BLOCK_DIM, (m_ + BLOCK_DIM - 1) / BLOCK_DIM);

    // record start event
    CUDA_CALL(cudaEventRecord(start, stream_));

    // launch kernel
    wmma_mm<<<blocks_per_grid, threads_per_block, 0, stream_>>>(m_, k_, n_, d_x, d_y, d_out);
    CUDA_CALL(cudaGetLastError());

    // record stop event
    CUDA_CALL(cudaEventRecord(stop, stream_));
}

template <typename T>
std::vector<T>
FastMM<T>::get()
{
    std::vector<T> out(m_ * n_);
    CUDA_CALL(cudaMemcpyAsync(out.data(), d_out, m_ * n_ * sizeof(T), cudaMemcpyDeviceToHost, stream_));
    CUDA_CALL(cudaStreamSynchronize(stream_));
    return out;
}

template <typename T>
float
FastMM<T>::time()
{
    float ms;
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

template class FastMM<float>;
template class FastMM<double>;