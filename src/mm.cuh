#include <vector>

__global__ void
mm(size_t m, size_t n, size_t p, float *x, float *y, float *out);

/**
 * @class MM
 * @brief A class to perform the MM operation using CUDA.
 *
 * The MM operation is defined as XY, where X has dimensions m x n and Y has
 * dimensions n x p. The result of the operation is a matrix of dimensions m x p.
 */
class MM
{
public:
    /**
     * @brief Constructs a MM object.
     * @param x The first matrix.
     * @param y The second matrix.
     * @param m The number of rows in the first matrix.
     * @param n The number of columns in the first matrix and the number of rows in the second matrix.
     * @param p The number of columns in the second matrix.
     */
    MM(const std::vector<float> &x, const std::vector<float> &y, const size_t m, const size_t n, const size_t p);

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
     * @return A vector containing the result.
     */
    std::vector<float>
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
    const size_t p_;
    const std::vector<float> &x_;
    const std::vector<float> &y_;

    float *d_x;
    float *d_y;
    float *d_out;

    cudaStream_t stream_;
    cudaEvent_t start;
    cudaEvent_t stop;
};
