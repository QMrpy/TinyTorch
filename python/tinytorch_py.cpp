#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <sstream>

#include "../src/Tensor.hpp"

namespace py = pybind11;

void init_module(py::module& m) {
    py::class_<Tensor>(m, "Tensor")
    .def(py::init<size_t>())
    .def(py::init<std::vector<size_t>&>())
    .def(py::init<float*, size_t>())
    .def(py::init<float*, size_t, bool>())
    .def(py::init<float*, std::vector<size_t>&>())
    .def(py::init<float*, std::vector<size_t>&, bool>())
    .def("__add__", [](Tensor& t1, Tensor& t2) -> Tensor {
        return t1 + t2;
    })
    .def("__sub__", [](Tensor& t1, Tensor& t2) -> Tensor {
        return t1 - t2;
    })
    .def("__mul__", [](Tensor& t, float o) -> Tensor {
        return t * o;
    })
    .def("__mul__", [](float o, Tensor& t) -> Tensor {
        return o * t;
    })
    .def("__mul__", [](Tensor& t1, Tensor& t2) -> Tensor {
        return t1 * t2;
    })
    .def("size", &Tensor::size)
    .def("shape", &Tensor::shape)
    .def("requires_grad_", &Tensor::set_requires_grad)
    .def("requires_grad", &Tensor::requires_grad)
    .def("data", &Tensor::data)
    .def("grad", &Tensor::gradient)
    .def("backward", &Tensor::backward)
    .def("__repr__", [](Tensor& t) {
        std::stringstream buffer;
        std::streambuf* buf = std::cout.rdbuf(buffer.rdbuf());
        std::cout << t << std::endl;

        return buffer.str();
    });
}

PYBIND11_MODULE(tinytorch_py, m) {
    init_module(m);
} 