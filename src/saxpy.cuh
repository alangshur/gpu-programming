#include <vector>

__global__ void
saxpy(size_t n, float a, float *x, float *y, float *out);

/**
 * @class SAXPY
 * @brief A class to perform the SAXPY operation using CUDA.
 *
 * The SAXPY operation is defined as a * X + Y where X and Y are vectors of
 * length n and a is a scalar.
 */
class SAXPY
{
public:
    /**
     * @brief Constructs a SAXPY object.
     * @param a Scalar multiplier.
     * @param x The first vector.
     * @param y The second vector.
     * @param n The length of the vectors.
     */
    SAXPY(const float a, const std::vector<float> &x, const std::vector<float> &y, const size_t n);

    /**
     * @brief Destroys the SAXPY object.
     */
    ~SAXPY();

    /**
     * @brief Runs the SAXPY operation.
     */
    void
    run();

    /**
     * @brief Retrieves the result of the SAXPY operation.
     * @return A vector containing the result.
     */
    std::vector<float>
    get();

    /**
     * @brief Retrieves the time taken to run the SAXPY operation.
     * @return The time taken in milliseconds.
     */
    float
    time();

private:
    const float a_;
    const std::vector<float> &x_;
    const std::vector<float> &y_;
    const size_t n_;

    float *d_x;
    float *d_y;
    float *d_out;

    cudaStream_t stream_;
    cudaEvent_t start;
    cudaEvent_t stop;
};
