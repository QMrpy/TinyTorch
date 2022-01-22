cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_STANDARD 17)

project(tinytorch_py)

find_package(pybind11 REQUIRED)

include_directories(../src/)

set(CXX_SOURCES "../src/Tensor.cpp")
set(CXX_HEADERS "../src/Tensor.hpp")
set(PYTHON_SOURCES "./tinytorch_py.cpp")

pybind11_add_module(tinytorch_py
    ${CXX_SOURCES}
    ${CXX_HEADERS}
    ${PYTHON_SOURCES})

target_link_libraries(tinytorch_py PUBLIC)

install(TARGETS tinytorch_py
    COMPONENT python
    LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}")