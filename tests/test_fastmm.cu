#include <gtest/gtest.h>
#include <iostream>

#include "../src/utils/mat.cuh"
#include "../src/utils/type.cuh"
#include "../src/fastmm.cuh"

TEST(FastMMTest, FastMMSmall)
{
    const size_t m = 10;
    const size_t n = 10;
    const size_t k = 10;

    std::vector<__half> x = vec_to_half(create_incremental_mat(m, k));
    std::vector<__half> y = vec_to_half(create_incremental_mat(k, n));

    FastMM<__half, float> fastmm(x, y, m, n, k);
    fastmm.run();

    std::vector<float> out = fastmm.get();
    EXPECT_EQ(out.size(), m * n);

    print_mat(x, m, k);
    print_mat(y, k, n);
    print_mat(out, m, n);

    GTEST_LOG_(INFO) << "Time taken: " << fastmm.time() << " ms";
}

TEST(FastMMTest, FastMMLarge)
{
    const size_t m = 1024;
    const size_t n = 1024;
    const size_t k = 1024;

    std::vector<__half> x = vec_to_half(create_random_mat(m, k));
    std::vector<__half> y = vec_to_half(create_random_mat(k, n));

    FastMM<__half, float> fastmm(x, y, m, n, k);
    fastmm.run();

    std::vector<float> out = fastmm.get();
    EXPECT_EQ(out.size(), m * n);

    GTEST_LOG_(INFO) << "Time taken: " << fastmm.time() << " ms";
}

int
main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
