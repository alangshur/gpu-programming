#include <iostream>
#include <vector>
#include <string>

#include "utils/mat.cuh"
#include "mm.cuh"
#include "saxpy.cuh"

void
run_mm()
{
    const size_t M = 1000;
    const size_t N = 2000;
    const size_t P = 1000;

    std::vector<float> x = create_random_mat(M, N, 0.0f, 1.0f);
    std::vector<float> y = create_random_mat(N, P, 0.0f, 1.0f);

    MM mm1(x, y, M, N, P);
    MM mm2(x, y, M, N, P);

    mm1.run();
    mm2.run();

    std::vector<float> out1 = mm1.get();
    std::vector<float> out2 = mm2.get();

    std::cout << "out1[0] = " << out1[0] << "(time: " << mm1.time() << "ms)" << std::endl;
    std::cout << "out2[0] = " << out2[0] << "(time: " << mm2.time() << "ms)" << std::endl;
}

void
run_saxpy()
{
    const size_t N = 1000;
    std::vector<float> x = create_random_vec(N, 0.0f, 1.0f);
    std::vector<float> y = create_random_vec(N, 0.0f, 1.0f);

    SAXPY saxpy1(2.0f, x, y, N);
    SAXPY saxpy2(3.0f, x, y, N);

    saxpy1.run();
    saxpy2.run();

    std::vector<float> out1 = saxpy1.get();
    std::vector<float> out2 = saxpy2.get();

    std::cout << "out1[0] = " << out1[0] << "(time: " << saxpy1.time() << "ms)" << std::endl;
    std::cout << "out2[0] = " << out2[0] << "(time: " << saxpy2.time() << "ms)" << std::endl;
}

int
main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <mm|saxpy>" << std::endl;
        return 1;
    }

    std::string arg = argv[1];
    if (arg == "mm")
    {
        run_mm();
    }
    else if (arg == "saxpy")
    {
        run_saxpy();
    }
    else
    {
        std::cerr << "Invalid argument: " << arg << std::endl;
        std::cerr << "Usage: " << argv[0] << " <mm|saxpy>" << std::endl;
        return 1;
    }

    return 0;
}
