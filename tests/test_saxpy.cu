#include <gtest/gtest.h>
#include <iostream>

#include "../src/utils/mat.cuh"
#include "../src/saxpy.cuh"

TEST(SAXPYTest, SimpleTest)
{
    const size_t N = 1000;
    std::vector<float> x = create_random_vec<float>(N, 0.0f, 1.0f);
    std::vector<float> y = create_random_vec<float>(N, 0.0f, 1.0f);

    SAXPY saxpy(2.0f, x, y, N);
    saxpy.run();

    std::vector<float> out = saxpy.get();
    EXPECT_EQ(out.size(), N);

    GTEST_LOG_(INFO) << "Time taken: " << saxpy.time() << " ms";
}

int
main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
