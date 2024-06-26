#include <vector>

template <typename T>
__global__ void
mm(size_t m, size_t n, size_t k, T *x, T *y, T *out);

/**
 * @class MM
 * @brief A class to perform the MM operation using CUDA.
 *
 * The MM operation is defined as XY, where X has dimensions m x k and Y has
 * dimensions k x n. The result of the operation is a matrix of dimensions m x n.
 */
template <typename T>
class MM
{
public:
    /**
     * @brief Constructs a MM object.
     * @param x The first matrix, stored in row-major order.
     * @param y The second matrix, stored in row-major order.
     * @param m The number of rows in the first matrix.
     * @param n The number of columns in the second matrix.
     * @param k The number of columns in the first matrix and the number of rows in the second matrix.
     */
    MM(const std::vector<T> &x, const std::vector<T> &y, const size_t m, const size_t n, const size_t k);

    /**
     * @brief Destroys the MM object.
     */
    ~MM();

    /**
     * @brief Runs the MM operation.
     */
    void
    run();

    /**
     * @brief Retrieves the result of the MM operation.
     * @return A vector containing the result, stored in row-major order.
     */
    std::vector<T>
    get();

    /**
     * @brief Retrieves the time taken to run the MM operation.
     * @return The time taken in milliseconds.
     */
    float
    time();

private:
    const size_t m_;
    const size_t n_;
    const size_t k_;
    const std::vector<T> &x_;
    const std::vector<T> &y_;

    T *d_x;
    T *d_y;
    T *d_out;

    cudaStream_t stream_;
    cudaEvent_t start;
    cudaEvent_t stop;
};
