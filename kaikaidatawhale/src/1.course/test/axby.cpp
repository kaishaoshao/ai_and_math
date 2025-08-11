//
//
//
#include <armadillo>
#include <glog/logging.h>
#include <gtest/gtest.h>

void Axby(const arma::fmat &x, const arma::fmat &w, const arma::fmat &b,
          arma::fmat &y) {
    // 把代码写这里 完成y = w * x + b的运算
    if(w.n_cols !=x.n_rows || w.n_rows != b.n_rows || b.n_cols != x.n_cols){
        LOG(ERROR) << "The size of w, x, b is not correct!";
        return;
    }
    // 计算y = w * x + b
    y = w * x + b;
}

TEST(test_arma, Axby) {
    using namespace arma;
    fmat w = "1,2,3;"
             "4,5,6;"
             "7,8,9;";
  
    fmat x = "1,2,3;"
             "4,5,6;"
             "7,8,9;";
  
    fmat b = "1,1,1;"
             "2,2,2;"
             "3,3,3;";
  
    fmat answer = "31,37,43;"
                  "68,83,98;"
                  "105,129,153";
  
    fmat y;
    Axby(x, w, b, y);
    ASSERT_EQ(approx_equal(y, answer, "absdiff", 1e-5f), true);
}

void EPowerMinus(const arma::fmat &x, arma::fmat &y){
    // 把代码写这里 完成y = e^{-x}的运算
    y = exp(-x);
}

TEST(test_arma, e_power_minus) {
    using namespace arma;

    // 创建一个224x224的矩阵x，并用随机数填充
    fmat x(224, 224);
    x.randu();

    // 创建一个空矩阵y
    fmat y;
    // 调用EPowerMinus函数，计算y = e^{-x}
    EPowerMinus(x, y);

    // 将矩阵x的数据存储到std::vector中
    // x.mem 是指向矩阵 x 数据的指针
    // x.mem + 224 * 224 是指向矩阵 x 数据末尾的指针
    std::vector<float> x1(x.mem, x.mem + 224 * 224);

    // 检查y是否为空
    ASSERT_EQ(y.empty(), false);

    // 遍历矩阵的每个元素，检查y中的值是否等于e^{-x}，误差在1e-5以内
    for(int i = 0; i < 224 * 224; i++){
        ASSERT_LE(std::abs(std::exp(-x1.at(i)) - y.at(i)), 1e-5f);
    }
}

void Axpy(const arma::fmat &x, arma::fmat &Y,float a, float y){
    // 编写Y = a * x + y
    Y = a * x + y;
}

TEST(test_arma, axpy){
    using namespace arma;
    // 创建一个224x224的矩阵x，并用随机数填充
    fmat x(224, 224);
    x.randu();

    // 创建一个空矩阵Y
    fmat Y;
    // 创建两个浮点数a和y
    float a = 3.f;
    float y = 4.f;
    // 调用Axpy函数，计算Y = a * x + y
    Axpy(x, Y, a, y);
    // 检查Y是否为空
    ASSERT_EQ(Y.empty(), false);
    // 检查Y中的每个元素是否等于a * x + y
    // std::vector<float> x1(Y.mem, Y.mem + 224 * 224);    
    for(int i = 0; i < 224 * 224; i++){
        ASSERT_LE(std::abs(a * x.at(i) + y - Y.at(i)), 1e-5f);
    }
}