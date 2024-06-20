#include <gtest/gtest.h>
#include <iostream>

#include "../src/utils/mat.cuh"
#include "../src/mm.cuh"

TEST(MMTest, SmallMM)
{
    const size_t M = 10;
    const size_t N = 20;
    const size_t P = 10;

    std::vector<float> x = create_random_mat<float>(M, N, 0.0f, 1.0f);
    std::vector<float> y = create_random_mat<float>(N, P, 0.0f, 1.0f);

    MM<float> mm(x, y, M, N, P);
    mm.run();

    std::vector<float> out = mm.get();
    EXPECT_EQ(out.size(), M * P);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

TEST(MMTest, LargeMMFloat)
{
    const size_t M = 1024;
    const size_t N = 1024;
    const size_t P = 1024;

    std::vector<float> x = create_random_mat<float>(M, N, 0.0f, 1.0f);
    std::vector<float> y = create_random_mat<float>(N, P, 0.0f, 1.0f);

    MM<float> mm(x, y, M, N, P);
    mm.run();

    std::vector<float> out = mm.get();
    EXPECT_EQ(out.size(), M * P);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

TEST(MMTest, LargeMMDouble)
{
    const size_t M = 1024;
    const size_t N = 1024;
    const size_t P = 1024;

    std::vector<double> x = create_random_mat<double>(M, N, 0.0f, 1.0f);
    std::vector<double> y = create_random_mat<double>(N, P, 0.0f, 1.0f);

    MM<double> mm(x, y, M, N, P);
    mm.run();

    std::vector<double> out = mm.get();
    EXPECT_EQ(out.size(), M * P);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

int
main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
