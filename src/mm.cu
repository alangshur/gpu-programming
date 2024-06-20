#include <vector>
#include <algorithm>

#include "utils/cuda.cuh"
#include "mm.cuh"

#define BLOCK_DIM 32

template <typename T>
__global__ void
mm(size_t m, size_t n, size_t p, T *x, T *y, T *out)
{
    __shared__ T x_tile[BLOCK_DIM][BLOCK_DIM];
    __shared__ T y_tile[BLOCK_DIM][BLOCK_DIM];

    size_t row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    size_t tile_row = threadIdx.y;
    size_t tile_col = threadIdx.x;

    T acc = 0.0;

    // iterate over all X and Y tiles to compute OUT tile
    size_t num_tiles = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx)
    {
        // load tile from X into shared memory
        size_t x_tile_row = blockIdx.y * BLOCK_DIM + threadIdx.y;
        size_t x_tile_col = tile_idx * BLOCK_DIM + threadIdx.x;
        if (x_tile_row < m && x_tile_col < n)
        {
            x_tile[tile_row][tile_col] = x[x_tile_row * n + x_tile_col];
        }
        else
        {
            x_tile[tile_row][tile_col] = 0.0;
        }

        // load tile from Y into shared memory
        size_t y_tile_row = tile_idx * BLOCK_DIM + threadIdx.y;
        size_t y_tile_col = blockIdx.x * BLOCK_DIM + threadIdx.x;
        if (y_tile_row < n && y_tile_col < p)
        {
            y_tile[tile_row][tile_col] = y[y_tile_row * p + y_tile_col];
        }
        else
        {
            y_tile[tile_row][tile_col] = 0.0;
        }

        // sync threads so both tiles are loaded in shared memory
        __syncthreads();

        // compute matrix multiplication between tiles
        for (size_t k = 0; k < BLOCK_DIM; ++k)
        {
            acc += x_tile[tile_row][k] * y_tile[k][tile_col];
        }

        // sync threads so all threads can reuse shared memory for next tiles
        __syncthreads();
    }

    out[row * p + col] = acc;
}

template <typename T>
MM<T>::MM(const std::vector<T> &x, const std::vector<T> &y, const size_t m, const size_t n, const size_t p)
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
MM<T>::~MM()
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
MM<T>::run()
{
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid((p_ + BLOCK_DIM - 1) / BLOCK_DIM, (m_ + BLOCK_DIM - 1) / BLOCK_DIM);

    // record start event
    CUDA_CALL(cudaEventRecord(start, stream_));

    // launch kernel
    mm<<<blocks_per_grid, threads_per_block, 0, stream_>>>(m_, n_, p_, d_x, d_y, d_out);
    CUDA_CALL(cudaGetLastError());

    // record stop event
    CUDA_CALL(cudaEventRecord(stop, stream_));
}

template <typename T>
std::vector<T>
MM<T>::get()
{
    std::vector<T> out(m_ * p_);
    CUDA_CALL(cudaMemcpyAsync(out.data(), d_out, m_ * p_ * sizeof(T), cudaMemcpyDeviceToHost, stream_));
    CUDA_CALL(cudaStreamSynchronize(stream_));
    return out;
}

template <typename T>
float
MM<T>::time()
{
    float ms;
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

template class MM<float>;
template class MM<double>;