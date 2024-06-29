#include <gtest/gtest.h>
#include <iostream>

#include "../src/utils/mat.cuh"
#include "../src/utils/type.cuh"
#include "../src/mm.cuh"

TEST(MMTest, SmallMM)
{
    const size_t m = 10;
    const size_t n = 10;
    const size_t k = 10;

    std::vector<float> x = create_mat(m, k, 1.0f);
    std::vector<float> y = create_mat(k, n, 1.0f);

    MM<float> mm(x, y, m, n, k);
    mm.run();

    std::vector<float> out = mm.get();
    EXPECT_EQ(out.size(), m * n);

    print_mat(x, m, k);
    print_mat(y, k, n);
    print_mat(out, m, n);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

TEST(MMTest, LargeMMFloat)
{
    const size_t m = 1024;
    const size_t n = 1024;
    const size_t k = 1024;

    std::vector<float> x = create_random_mat(m, k);
    std::vector<float> y = create_random_mat(k, n);

    MM<float> mm(x, y, m, n, k);
    mm.run();

    std::vector<float> out = mm.get();
    EXPECT_EQ(out.size(), m * n);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

TEST(MMTest, LargeMMDouble)
{
    const size_t m = 1024;
    const size_t n = 1024;
    const size_t k = 1024;

    std::vector<double> x = vec_to_double(create_random_mat(m, k, 0.0f, 1.0f));
    std::vector<double> y = vec_to_double(create_random_mat(k, n, 0.0f, 1.0f));

    MM<double> mm(x, y, m, n, k);
    mm.run();

    std::vector<double> out = mm.get();
    EXPECT_EQ(out.size(), m * n);

    GTEST_LOG_(INFO) << "Time taken: " << mm.time() << " ms";
}

int
main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
