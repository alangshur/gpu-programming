#include <vector>
#include <algorithm>

#include "utils/cuda.cuh"
#include "saxpy.cuh"

__global__ void
saxpy(size_t n, float a, float *x, float *y, float *out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a * x[i] + y[i];
    }
}

SAXPY::SAXPY(const float a, const std::vector<float> &x, const std::vector<float> &y, const size_t n)
    : a_(a), x_(x), y_(y), n_(n)
{
    // verify vector sizes
    if (x.size() != n_ || x.size() != y.size())
        throw std::runtime_error("Vector sizes do not match");
    else if (x.size() == 0 || y.size() == 0)
        throw std::runtime_error("Vector size cannot be zero");

    // create stream
    CUDA_CALL(cudaStreamCreate(&stream_));

    // create events
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    // allocate device memory
    CUDA_CALL(cudaMallocAsync(&d_x, n_ * sizeof(float), stream_));
    CUDA_CALL(cudaMallocAsync(&d_y, n_ * sizeof(float), stream_));
    CUDA_CALL(cudaMallocAsync(&d_out, n_ * sizeof(float), stream_));

    // copy data to device
    CUDA_CALL(cudaMemcpyAsync(d_x, x.data(), n_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
    CUDA_CALL(cudaMemcpyAsync(d_y, y.data(), n_ * sizeof(float), cudaMemcpyHostToDevice, stream_));
}

SAXPY::~SAXPY()
{
    cudaStreamSynchronize(stream_);

    // destroy stream
    cudaStreamDestroy(stream_);

    // free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);
}

void
SAXPY::run()
{
    const size_t num_threads = std::min<size_t>(256, n_);
    const size_t num_blocks = (n_ + num_threads - 1) / num_threads;

    // record start event
    CUDA_CALL(cudaEventRecord(start, stream_));

    // launch kernel
    saxpy<<<num_blocks, num_threads, 0, stream_>>>(n_, a_, d_x, d_y, d_out);
    CUDA_CALL(cudaGetLastError());

    // record stop event
    CUDA_CALL(cudaEventRecord(stop, stream_));
}

std::vector<float>
SAXPY::get()
{
    std::vector<float> out(n_);
    CUDA_CALL(cudaMemcpyAsync(out.data(), d_out, n_ * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CALL(cudaStreamSynchronize(stream_));
    return out;
}

float
SAXPY::time()
{
    float ms;
    CUDA_CALL(cudaEventSynchronize(stop));
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}
