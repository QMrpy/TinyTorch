#include "Tensor.hpp"

Tensor::Tensor(float* data) : __data(data) {
    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    __size = sizeof(__data) / sizeof(float);
    __shape.push_back(__size);
}

Tensor::Tensor(float* data, bool requires_grad) : __data(data), __requires_grad(requires_grad) {
    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    __size = sizeof(__data) / sizeof(float);
    __shape.push_back(__size);

    if (__requires_grad) {
        float grad = 1.0;
        __grad = &grad;
    }
}

Tensor::Tensor(float* data, std::vector<size_t>& shape) : __data(data) {
    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    __size = sizeof(__data) / sizeof(float);

    int tsize = 1;
    for (int i = 0; i < shape.size(); i++)
        tsize *= shape[i];
    
    if (__size != tsize) 
        throw std::invalid_argument("Shape and Size are inconsistent for given Tensor.");

    __shape = shape;
}

Tensor::Tensor(float* data, std::vector<size_t>& shape, bool requires_grad) 
    : __data(data), __requires_grad(requires_grad) {

    if (__data == nullptr)
        throw std::runtime_error("Supplied data to Tensor is null.");

    __size = sizeof(__data) / sizeof(float);

    int tsize = 1;
    for (int i = 0; i < shape.size(); i++)
        tsize *= shape[i];
    
    if (__size != tsize) 
        throw std::invalid_argument("Shape and Size are inconsistent for given Tensor.");

    __shape = shape;

    if (__requires_grad) {
        float grad = 1.0;
        __grad = &grad;
    }
}

size_t Tensor::size() {
    return __size;
}

float* Tensor::data() {
    return __data;
}

std::vector<size_t> Tensor::shape() {
    return __shape;
}

Tensor& Tensor::operator+= (Tensor& t) {
    if ((*this).data() == nullptr || t.data() == nullptr)
        throw std::runtime_error("Received null operand while trying to perform addition.");

    if ((*this).shape() != t.shape())
        throw std::invalid_argument("Tensor shape mismatch while trying to perform addition.");

    for (int i = 0; i < (*this).size(); i++)
        __data[i] += t.data()[i];

    return *this;
}

Tensor& Tensor::operator-= (Tensor& t) {
    if ((*this).data() == nullptr || t.data())
        throw std::runtime_error("Received null operand while trying to perform subtraction.");

    if ((*this).shape() != t.shape())
        throw std::invalid_argument("Tensor shape mismatch while trying to perform subtraction.");

    for (int i = 0; i < (*this).size(); i++)
        __data[i] -= t.data()[i];

    return *this;
}

Tensor& Tensor::operator*= (float o) {
    if ((*this).data() == nullptr)
        throw std::runtime_error("Received null operand while trying to perform scalar multiplication.");

    for (int i = 0; i < (*this).size(); i++)
        __data[i] *= o;

    return *this;
}

void Tensor::set_requires_grad() {
    float grad = 1.0;
    __grad = &grad;

    __requires_grad = true;
}

bool Tensor::requires_grad() {
    return __requires_grad;
}

float* Tensor::gradient() {
    return __grad;
}

/*float* Tensor::operator* (float* o) {
    for (int i = 0; i < sizeof(o) / sizeof(float); i++)
        this->__grad[i] *= o[i];
    return this->__grad;
}

void Tensor::backward() {
    if (!(*this).requires_grad())
        throw std::invalid_argument("Cannot compute gradients on a Tensor with requires_grad=false.");

    for (auto t: (*this).__nodes) {
        //t.__grad = (*this).gradient() * t.gradient();
        float* a = (*this).gradient();
        float* b = t.gradient();
        float c = a * b;
        t.__grad = &c;
        t.backward();
    }
}*/
