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
