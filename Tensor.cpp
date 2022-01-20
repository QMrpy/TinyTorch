#include "Tensor.hpp"

Tensor::Tensor(size_t size) : __size(size) {
    if (__size <= 0)
        throw std::runtime_error("Cannot create a Tensor with given size.");

    __data = new float[__size];
    std::fill(__data, __data + __size, 0.0);

    __shape.push_back(__size);
}

Tensor::Tensor(std::vector<size_t>& shape) : __shape(shape) {
    if (__shape.empty())
        throw std::runtime_error("Cannot create a Tensor with given shape.");

    __size = 1;
    for (int i = 0; i < __shape.size(); i++) {
        if (__shape[i] <= 0)
            throw std::runtime_error("Got impossible size for shape of Tensor.");
        __size *= __shape[i];
    }

    __data = new float[__size];
    std::fill(__data, __data + __size, 0.0);
}

Tensor::Tensor(float* data, size_t size) : __data(data), __size(size) {
    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    if (__size <= 0)
        throw std::runtime_error("Cannot create a Tensor with given size.");

    __shape.push_back(__size);
}

Tensor::Tensor(float* data, size_t size, bool requires_grad) 
    : __data(data), __size(size), __requires_grad(requires_grad) {

    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    if (__size <= 0)
        throw std::runtime_error("Cannot create a Tensor with given size.");

    __shape.push_back(__size);

    if (__requires_grad) {
       __grad = new float[__size];
       std::fill(__grad, __grad + __size, 0.0);
    }
}

Tensor::Tensor(float* data, std::vector<size_t>& shape) : __data(data), __shape(shape) {
    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    if (__shape.empty())
        throw std::runtime_error("Cannot create a Tensor with given shape.");

    __size = 1;
    for (int i = 0; i < __shape.size(); i++) {
        if (__shape[i] <= 0)
            throw std::runtime_error("Got impossible size for shape of Tensor.");
        __size *= __shape[i];
    }
}

Tensor::Tensor(float* data, std::vector<size_t>& shape, bool requires_grad) 
    : __data(data), __shape(shape), __requires_grad(requires_grad) {

    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    if (__shape.empty())
        throw std::runtime_error("Cannot create a Tensor with given shape.");

    __size = 1;
    for (int i = 0; i < __shape.size(); i++) {
        if (__shape[i] <= 0)
            throw std::runtime_error("Got impossible size for shape of Tensor.");
        __size *= __shape[i];
    }

    if (__requires_grad) {
       __grad = new float[__size];
       std::fill(__grad, __grad + __size, 0.0);
    }
}

Tensor& Tensor::operator+ (Tensor& t) {
    if (t.data() == nullptr)
        throw std::runtime_error("Received null operand while trying to perform addition.");

    if (__shape != t.shape())
        throw std::runtime_error("Tensor shape mismatch while trying to perform addition.");

    Tensor* res = new Tensor(__shape);
    for (int i = 0; i < __size; i++)
        res->__data[i] = __data[i] + t.data()[i];

    if (__requires_grad && t.requires_grad()) {
        res->set_requires_grad();
        res->__nodes.push_back(this);
        res->__nodes.push_back(&t);

        auto lambda = [&, res] () {
            for (int i = 0; i < __size; i++) {
                __grad[i] += res->__grad[i];
                t.__grad[i] += res->__grad[i];
            }
        };

        res->__backward = lambda;
    }

    return *res;
}

Tensor& Tensor::operator- (Tensor& t) {
    return *this + t * -1.0f;
}

Tensor& Tensor::operator* (float o) {
    Tensor* res = new Tensor(__shape);
    for (int i = 0; i < __size; i++)
        res->__data[i] = o * __data[i];

    if (__requires_grad) {
        res->set_requires_grad();
        res->__nodes.push_back(this);

        auto lambda = [&, res] () {
            for (int i = 0; i < __size; i++) {
                __grad[i] += o * res->__grad[i];
            }
        };

        res->__backward = lambda;
    }

    return *res;
}

Tensor& Tensor::operator* (Tensor& t) {
    if (t.data() == nullptr)
        throw std::runtime_error("Received null operand while trying to perform multiplication.");

    if (__shape != t.shape())
        throw std::runtime_error("Tensor shape mismatch while trying to perform multiplication.");

    Tensor* res = new Tensor(__shape);
    for (int i = 0; i < __size; i++)
        res->__data[i] = __data[i] * t.data()[i];

    if (__requires_grad && t.requires_grad()) {
        res->set_requires_grad();
        res->__nodes.push_back(this);
        res->__nodes.push_back(&t);

        auto lambda = [&, res] () {
            for (int i = 0; i < __size; i++) {
                __grad[i] += t.__data[i] * res->__grad[i];
                t.__grad[i] += __data[i] * res->__grad[i];
            }
        };

        res->__backward = lambda;
    }
    
    return *res;
}

size_t Tensor::size() {
    return __size;
}

std::vector<size_t> Tensor::shape() {
    return __shape;
}

void Tensor::set_requires_grad() {
    __grad = new float[__size];
    std::fill(__grad, __grad + __size, 0.0);

    __requires_grad = true;
}

bool Tensor::requires_grad() {
    return __requires_grad;
}

float* Tensor::data() {
    return __data;
}

float* Tensor::gradient() {
    return __grad;
}

void Tensor::backward() {
    if (!__requires_grad)
        throw std::runtime_error("Cannot compute gradients on a Tensor with requires_grad=false.");

    std::vector<Tensor*> topological_order;
    __topological_sort(this, topological_order);

    std::reverse(topological_order.begin(), topological_order.end());
    std::fill(this->__grad, this->__grad + this->__size, 1.0);

    for (auto t: topological_order) {
        if (t->__backward)
            t->__backward();
    }
}

std::ostream& operator<< (std::ostream& stream, Tensor& t) {
    stream << std::endl;

    stream << "Tensor(data=[";
    for (int i = 0; i < t.size(); i++) 
        stream << t.data()[i] << ",";

    stream << "], gradient=[";
    if (t.requires_grad()) {
        for (int i = 0; i < t.size(); i++) 
            stream << t.gradient()[i] << ",";
        stream << "])" << std::endl;
    } else {
        stream << "None])" << std::endl;
    }

    return stream;
}

void Tensor::__topological_sort(Tensor* t, std::vector<Tensor*>& topological_order) {
    if (!t->__visited) {
        t->__visited = true;
        for (auto node: t->__nodes) {
            __topological_sort(node, topological_order);
        }

        topological_order.push_back(t);
    }
}

