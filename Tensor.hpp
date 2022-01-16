#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>

class Tensor {
    public:
        Tensor() = delete;
        Tensor(float* data);
        Tensor(float* data, bool requires_grad);
        Tensor(float* data, std::vector<size_t>& shape);
        Tensor(float* data, std::vector<size_t>& shape, bool requires_grad);
        Tensor& operator= (Tensor& t);
        Tensor& operator+= (Tensor& t);
        Tensor& operator-= (Tensor& t);
        Tensor& operator*= (float o);
        Tensor& operator*= (Tensor& t);
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

    private:
        float* __data;
        float* __grad;
        bool __requires_grad = false;
        size_t __size;
        std::vector<size_t> __shape;
        std::vector<Tensor> __nodes;

        float* operator+ (float* o);
        float* operator- (float* o);
        float* operator* (float* o);
};

