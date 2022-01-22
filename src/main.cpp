#include "Tensor.hpp"

int main() {
    float a_data[12] = {1.0f, 1.0f, 7.0f, 9.0f, 2.0f, 3.0f, 6.0f, 87.0f, 3.0f, 5.0f, 3.0f, 7.0f};
    std::vector<size_t> a_shape = {3, 4};

    Tensor* a = new Tensor(&a_data[0], a_shape);
    a->set_requires_grad();

    float b_data[12] = {6.0f, 2.0f, 4.0f, 3.0f, 5.0f, 7.8f, 9.3f, 65.4f, 2.3f, 7.6f, 5.4f, 7.8f};
    std::vector<size_t> b_shape = {3, 4};

    Tensor* b = new Tensor(&b_data[0], b_shape);
    b->set_requires_grad();

    std::cout << "Checking correctness of Tensor constructors." << std::endl;

    std::cout << *a << std::endl;
    std::cout << *b << std::endl; 

    std::cout << "Checking correctness of Tensor addition and multiplication operations." << std::endl;

    Tensor c = (*a) * (*a);
    std::cout << c << std::endl;

    Tensor d = (*b) * (*b) * 7.0f;
    std::cout << d << std::endl;

    Tensor e = (*b) * 4.0f;
    std::cout << e << std::endl;

    Tensor f = (*a) * (*b) * (*a);
    std::cout << f << std::endl;

    Tensor g = 5.0f * (*a);
    std::cout << g << std::endl;

    Tensor h = (*a) + (*b) + c + (d * d) + (e * f) + (g * 7.9f);
    std::cout << h << std::endl;

    std::cout << "Checking correctness of Tensor gradient operations." << std::endl;

    h.backward();

    std::cout << *a << std::endl;
    std::cout << *b << std::endl;
    std::cout << c << std::endl;
    std::cout << d << std::endl;
    std::cout << e << std::endl;
    std::cout << f << std::endl;
    std::cout << g << std::endl;
    std::cout << h << std::endl;

    return 0;
}