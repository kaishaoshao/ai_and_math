///
///
/// 

#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <armadillo>
#include <memory>
#include <vector>

namespace kaikai_infer {

// 通用模板类定义
template <typename T = float>
class Tensor {};

// uint8_t类型的特化模板类定义
template <>
class Tensor<uint8_t> {
    // 待实现
};

// float类型的特化模板类定义
template <>
class Tensor<float> {
public:
// 显示构造： explicit关键字用于防止隐式转换和复制初始化
// 这意味只能通过显式构造函数调用来创建对象，防止意外类型转换
// 默认构造函数： default关键字用于显示声明默认构造函数
// 通常用于需要默认构造的函数的类，但不需要自定义的构造函数逻辑
explicit Tensor() = default;

/**  创建张量
* @param channels: 张量的通道数
* @param rows: 张量的行数
* @param cols: 张量的列数
*/
explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols); 






};


}


#endif // __TENSOR_HPP__