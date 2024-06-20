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
*/

template <typename T1, typename T2, typename WMMA_M, typename WMMA_N, typename WMMA_K>
__global__ void
fastmm(size_t m, size_t n, size_t p, T1 *x, T1 *y, T1 *out)
{
    size_t warp_row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> acc_frag;

    // make sure accumulator starts from zero
    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    size_t num_frags = (n + WMMA_K - 1) / WMMA_K;
    for (size_t frag_index; frag_index < num_frags; ++frag_index)
    {
        size_t x_frag_row = blockIdx.y * WMMA_M;

        // load data into fragments
        nvcuda::wmma::load_matrix_sync(a_frag, x + blockIdx.y * WMMA_M * n + frag_index * WMMA_K, n);
        nvcuda::wmma::load_matrix_sync(b_frag, y + frag_index * WMMA_K * p + blockIdx.x * WMMA_K, p);

        // perform matrix multiplication
        nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
}

template <typename T>
FastMM<T>::FastMM(const std::vector<T> &x, const std::vector<T> &y, const size_t m, const size_t n, const size_t p)
    : m_(m), n_(n), p_(p), x_(x), y_(y)
{
    // verify vector sizes
    if (x.size() != m_ * n_ || y.size() != n_ * p_)
        throw std::runtime_error("Vector sizes do not match");
    else if (x.size() == 0 || y.size() == 0)
        throw std::runtime_error("Vector size cannot be zero");

    // create stream
    CUDA_CALL(cudaStreamCreate(&stream_));

    // create events
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    // allocate device memory
    CUDA_CALL(cudaMallocAsync(&d_x, m_ * n_ * sizeof(T), stream_));
    CUDA_CALL(cudaMallocAsync(&d_y, n_ * p_ * sizeof(T), stream_));
    CUDA_CALL(cudaMallocAsync(&d_out, m_ * p_ * sizeof(T), stream_));

    // copy data to device
    CUDA_CALL(cudaMemcpyAsync(d_x, x.data(), m_ * n_ * sizeof(T), cudaMemcpyHostToDevice, stream_));
    CUDA_CALL(cudaMemcpyAsync(d_y, y.data(), n_ * p_ * sizeof(T), cudaMemcpyHostToDevice, stream_));
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
    dim3 blocks_per_grid((p_ + BLOCK_DIM - 1) / BLOCK_DIM, (m_ + BLOCK_DIM - 1) / BLOCK_DIM);

    // record start event
    CUDA_CALL(cudaEventRecord(start, stream_));

    // launch kernel
    fastmm<<<blocks_per_grid, threads_per_block, 0, stream_>>>(m_, n_, p_, d_x, d_y, d_out);
    CUDA_CALL(cudaGetLastError());

    // record stop event
    CUDA_CALL(cudaEventRecord(stop, stream_));
}

template <typename T>
std::vector<T>
FastMM<T>::get()
{
    std::vector<T> out(m_ * p_);
    CUDA_CALL(cudaMemcpyAsync(out.data(), d_out, m_ * p_ * sizeof(T), cudaMemcpyDeviceToHost, stream_));
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