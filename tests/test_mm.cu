#include <gtest/gtest.h>
#include <iostream>

#include "../src/utils/mat.cuh"
#include "../src/mm.cuh"

TEST(MMTest, SmallMM)
{
    const size_t M = 10;
    const size_t N = 10;
    const size_t P = 10;

    std::vector<float> x = create_random_mat(M, N, 0.0f, 1.0f);
    std::vector<float> y = create_random_mat(N, P, 0.0f, 1.0f);

    MM mm(x, y, M, N, P);
    mm.run();

    std::vector<float> out = mm.get();
    EXPECT_EQ(out.size(), M * P);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

TEST(MMTest, LargeMM)
{
    const size_t M = 1024;
    const size_t N = 1024;
    const size_t P = 1024;

    std::vector<float> x = create_random_mat(M, N, 0.0f, 1.0f);
    std::vector<float> y = create_random_mat(N, P, 0.0f, 1.0f);

    MM mm(x, y, M, N, P);
    mm.run();

    std::vector<float> out = mm.get();
    EXPECT_EQ(out.size(), M * P);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

int
main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
