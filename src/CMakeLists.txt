cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_STANDARD 17)

project(TinyTorch)

set(tensor_hdrs "./Tensor.hpp")
set(tensor_srcs "./Tensor.cpp")

add_library(tensor
    ${tensor_srcs}
    ${tensor_hdrs})

add_executable(tinytorch "./main.cpp")

target_link_libraries(tinytorch
    tensor)