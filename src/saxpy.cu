#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include "utils/cuda.cuh"
#include "saxpy.cuh"

__global__ void
saxpy(int n, float a, float *x, float *y, float *out)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a * x[i] + y[i];
    }
}

SAXPY::SAXPY(float a, std::vector<float> &x, std::vector<float> &y) : a_(a), x_(x), y_(y), n_(x.size())
{
    // verify vector sizes
    if (x.size() != y.size() || x.size() == 0)
        throw std::runtime_error("Vector sizes do not match");

    // create stream
    CUDA_CALL(cudaStreamCreate(&stream_));

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
    const size_t num_threads = std::max<size_t>(256, n_);
    const size_t num_blocks = (n_ + num_threads - 1) / num_threads;
    saxpy<<<num_blocks, num_threads, 0, stream_>>>(n_, a_, d_x, d_y, d_out);
    CUDA_CALL(cudaGetLastError());
}

std::vector<float>
SAXPY::get()
{
    std::vector<float> out(n_);
    CUDA_CALL(cudaMemcpyAsync(out.data(), d_out, n_ * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CALL(cudaStreamSynchronize(stream_));
    return out;
}

int
main(void)
{
    const size_t N = 1000;
    std::vector<float> x(N);
    std::vector<float> y(N);
    for (size_t i = 0; i < N; ++i)
    {
        x[i] = 2.0f;
        y[i] = 3.0f;
    }

    SAXPY saxpy1(2.0f, x, y);
    SAXPY saxpy2(3.0f, x, y);

    saxpy1.run();
    saxpy2.run();

    std::vector<float> out1 = saxpy1.get();
    std::vector<float> out2 = saxpy2.get();

    std::cout << "out1[0] = " << out1[0] << std::endl;
    std::cout << "out2[0] = " << out2[0] << std::endl;

    return 0;
}