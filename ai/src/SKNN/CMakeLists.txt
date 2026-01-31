CMAKE_MINIMUM_REQUIRED_VERSION(3.10)
project(sknn LANGUAGES C)

add_library(sknn STATIC)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

install(TARGETS sknn)
