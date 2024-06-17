#include <vector>

__global__ void
saxpy(int n, float a, float *x, float *y, float *out);

/**
 * @class SAXPY
 * @brief A class to perform the SAXPY operation using CUDA.
 *
 * The SAXPY operation is defined as a * X + Y.
 */
class SAXPY
{
public:
    /**
     * @brief Constructs a SAXPY object.
     * @param a Scalar multiplier.
     * @param x Reference to the input vector X.
     * @param y Reference to the input vector Y.
     */
    SAXPY(float a, std::vector<float> &x, std::vector<float> &y);

    /**
     * @brief Destroys the SAXPY object.
     */
    ~SAXPY();

    /**
     * @brief Runs the SAXPY operation on the input vectors.
     */
    void
    run();

    /**
     * @brief Retrieves the result of the SAXPY operation.
     * @return A vector containing the result.
     */
    std::vector<float>
    get();

private:
    const size_t n_;
    const float a_;
    const std::vector<float> &x_;
    const std::vector<float> &y_;

    float *d_x;
    float *d_y;
    float *d_out;

    cudaStream_t stream_;
};
