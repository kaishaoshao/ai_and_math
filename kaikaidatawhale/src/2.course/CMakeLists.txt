cmake_minimum_required(VERSION 3.20)
project(kaikai_datawhale_course2)

# Google Benchmark库 用于基准测试
find_package(benchmark REQUIRED)
# GTest库 用于单元测试
find_package(GTest REQUIRED)
# glog库 用于日志记录
find_package(glog REQUIRED)
# OpenMP库 用于并行计算
find_package(OpenMP REQUIRED)
# Armadillo库 用于线性代数计算
find_package(Armadillo REQUIRED)
# BLAS库 用于线性代数计算
find_package(BLAS REQUIRED)
# LAPACK库 用于线性代数计算
find_package(LAPACK REQUIRED)

# 设置链接库 包含glog、GTest windows下还需要包含pthread
set(link_lib glog::glog GTest::gtest)
if(!WIN32)
    set(link_lib "${link_libraries} pthread")
endif()

# 设置链接数学库 包含Armadillo、BLAS、LAPACK
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

# 将./test目录下的所有文件添加到DIR_TEST_ARMA变量中
aux_source_directory(./test DIR_TEST_ARMA)
aux_source_directory(./source DIR_SOURCE_ARMA)

# 创建可执行文件kaikai_datawhale_course1 
add_executable(kaikai_datawhale_course2 main.cpp ${DIR_TEST_ARMA} ${DIR_SOURCE_ARMA})
# 链接可执行文件kaikai_datawhale_course1需要的库
target_link_libraries(kaikai_datawhale_course2 ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

# 设置可执行文件kaikai_datawhale_course1的头文件路径
target_include_directories(kaikai_datawhale_course2 PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(kaikai_datawhale_course2 PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(kaikai_datawhale_course2 PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(kaikai_datawhale_course2 PUBLIC ./include)

# 启用测试
enable_testing()