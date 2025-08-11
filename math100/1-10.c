#include <stdio.h>
#include <math.h>

// 问题1 ： f(x) = 2x^2 - 4x + 3, 求对称轴和顶点
void problem1() {
    printf(" -------- 问题1 -------\n");
    // 二次函数的系数
    float a = 2.0f, b = -4.0f, c = 3.0f;
    // 对称轴 x = -b / (2a)
    float axis_x = -b / (2.0f * a);
    // 顶点的 y 坐标
    float vertex_y = a * powf(axis_x, 2.0f) + b * axis_x + c;
    printf("函数 f(x) = 2x^2 - 4x + 3\n");
    printf("对称轴 x = %.2f, 顶点坐标 (%.2f, %.2f)\n\n", axis_x, axis_x, vertex_y);
}

// 问题2 ： 
void problem2() {

}

void problem3() {}

void problem4() {}

void problem5() {}

void problem6() {}

void problem7() {}

void problem8() {}

void problem9() {}

void problem10() {}

int main() {
    problem1();
    problem2();
    problem3();
    problem4();
    problem5();
    problem6();
    problem7();
    problem8();
    problem9();
    problem10();
  return 0;
}
