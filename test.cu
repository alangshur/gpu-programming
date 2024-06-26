#include <mma.h>
#include <cuda_runtime.h>

using namespace nvcuda;

__global__ void
wmma_example_kernel()
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> a_frag{};
}

int
main()
{
    wmma_example_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
