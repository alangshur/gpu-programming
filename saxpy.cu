#include <iostream>
#include <vector>

__global__ void saxpy(int n, float a, float *x, float *y, float *out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a * x[i] + y[i];
    }
}

int main(void) {
    const size_t N = 10;

    std::vector<float> x(N);
    std::vector<float> y(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_out;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    cudaMemcpy(d_x, x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<1, 10>>>(N, 2.0f, d_x, d_y, d_out);
    cudaDeviceSynchronize();

    std::vector<float> out(N);
    cudaMemcpy(out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "out: ";
    for (float i : out) std::cout << i << " ";
    std::cout << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_out);

    return 0;
}