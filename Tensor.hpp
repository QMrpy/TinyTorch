#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>
#include <functional>

class Tensor {
    public:
        Tensor() = delete;
        Tensor(size_t size);
        Tensor(std::vector<size_t>& shape);
        Tensor(float* data, size_t size);
        Tensor(float* data, size_t size, bool requires_grad);
        Tensor(float* data, std::vector<size_t>& shape);
        Tensor(float* data, std::vector<size_t>& shape, bool requires_grad);
        Tensor& operator+= (Tensor& t) = delete;
        Tensor& operator-= (Tensor& t) = delete;
        Tensor& operator*= (float o) = delete;
        Tensor& operator*= (Tensor& t) = delete;
        Tensor& operator+ (Tensor& t);
        Tensor& operator- (Tensor& t);
        Tensor& operator* (float o);
        Tensor& operator* (Tensor& t);
        size_t size();
        std::vector<size_t> shape();
        void set_requires_grad();
        bool requires_grad();
        float* data();
        float* gradient();
        void backward();
        friend std::ostream& operator<< (std::ostream& stream, Tensor& t);

    private:
        float* __data;
        float* __grad;
        bool __requires_grad = false;
        size_t __size;
        std::vector<size_t> __shape;
        std::vector<Tensor*> __nodes;
        std::function<void()> __backward;
        bool __visited = false;

        void __topological_sort(Tensor* t, std::vector<Tensor*>& topological_order);
};

