#include "Tensor.hpp"

int main() {
    float a_data[4] = {1.0f, 1.0f, 2.0f, 2.0f};
    std::vector<size_t> a_shape = {2, 2};

    Tensor* a = new Tensor(&a_data[0], a_shape);
    a->set_requires_grad();

    float b_data[4] = {6.0f, 2.0f, 4.0f, 3.0f};
    std::vector<size_t> b_shape = {2, 2};

    Tensor* b = new Tensor(&b_data[0], b_shape);

    std::cout << "Checking correctness of Tensor addition and multiplication operations." << std::endl;

    Tensor c = (*a) + (*b);
    std::cout << c << std::endl;

    Tensor d = (*a) - (*b);
    std::cout << d << std::endl;

    Tensor e = (*b) * 4;
    std::cout << e << std::endl;

    Tensor f = (*a) * (*b);
    std::cout << f << std::endl;

    Tensor g = (*a) * (*a);
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