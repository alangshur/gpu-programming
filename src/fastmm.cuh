#include <vector>

template <typename WMMA_T1, typename WMMA_T2>
__global__ void
wmma_mm(size_t m, size_t n, size_t k, WMMA_T1 *x, WMMA_T1 *y, WMMA_T2 *out);

/**
 * @class FastMM
 * @brief A class to perform the FastMM operation using CUDA.
 *
 * The FastMM operation is defined as XY, where X has dimensions m x k and Y has
 * dimensions k x n. The result of the operation is a matrix of dimensions m x n.
 */
template <typename T1, typename T2>
class FastMM
{
public:
    /**
     * @brief Constructs a FastMM object.
     * @param x The first matrix, stored in row-major order.
     * @param y The second matrix, stored in row-major order.
     * @param m The number of rows in the first matrix.
     * @param n The number of columns in the second matrix.
     * @param k The number of columns in the first matrix and the number of rows in the second matrix.
     */
    FastMM(const std::vector<T1> &x, const std::vector<T1> &y, const size_t m, const size_t n, const size_t k);

    /**
     * @brief Destroys the FastMM object.
     */
    ~FastMM();

    /**
     * @brief Runs the FastMM operation.
     */
    void
    run();

    /**
     * @brief Retrieves the result of the FastMM operation.
     * @return A vector containing the result, stored in row-major order.
     */
    std::vector<T2>
    get();

    /**
     * @brief Retrieves the time taken to run the FastMM operation.
     * @return The time taken in milliseconds.
     */
    float
    time();

private:
    const size_t m_;
    const size_t n_;
    const size_t k_;
    const std::vector<T1> &x_;
    const std::vector<T1> &y_;

    T1 *d_x;
    T1 *d_y;
    T2 *d_out;

    cudaStream_t stream_;
    cudaEvent_t start;
    cudaEvent_t stop;
};
