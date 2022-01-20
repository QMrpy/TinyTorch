#include "Tensor.hpp"

int main() {
    float a_data[12] = {1.0f, 1.0f, 7.0f, 9.0f, 2.0f, 3.0f, 6.0f, 87.0f, 3.0f, 5.0f, 3.0f, 7.0f};
    std::vector<size_t> a_shape = {3, 4};

    Tensor* a = new Tensor(&a_data[0], a_shape);
    a->set_requires_grad();

    float b_data[12] = {6.0f, 2.0f, 4.0f, 3.0f, 5.0f, 7.8f, 9.3f, 65.4f, 2.3f, 7.6f, 5.4f, 7.8f};
    std::vector<size_t> b_shape = {3, 4};

    Tensor* b = new Tensor(&b_data[0], b_shape);

    std::cout << "Checking correctness of Tensor constructors." << std::endl;

    std::cout << *a << std::endl;
    std::cout << *b << std::endl; 

    std::cout << "Checking correctness of Tensor addition and multiplication operations." << std::endl;

    Tensor c = (*a) + (*b);
    std::cout << c << std::endl;

    Tensor d = (*a) - (*b);
    std::cout << d << std::endl;

    Tensor e = (*b) * 4;
    std::cout << e << std::endl;

    Tensor f = (*a) * (*b);
    std::cout << f << std::endl;

    Tensor g = (*a) * (*a) * (*a);
    std::cout << g << std::endl;

    Tensor h = (*a) + g;
    std::cout << h << std::endl;

    std::cout << "Checking correctness of Tensor gradient operations." << std::endl;

    h.backward();

    std::cout << *a << std::endl;
    std::cout << g << std::endl;
    std::cout << h << std::endl;

    return 0;
}