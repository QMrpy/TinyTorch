#include "Tensor.hpp"

void print_tensor(Tensor& t) {
    std::cout << std::endl;

    std::cout << "Printing Tensor data: " << std::endl;
    for (int i = 0; i < t.size(); i++) 
        std::cout << t.data()[i] << " ";
    std::cout << std::endl;

    std::cout << "Printing Tensor gradient data: " << std::endl;
    for (int i = 0; i < t.size(); i++) 
        std::cout << t.gradient()[i] << " ";
    std::cout << std::endl;    
}

int main() {
    float a_data[4] = {1.0f, 1.0f, 2.0f, 2.0f};
    std::vector<size_t> a_shape = {2, 2};

    Tensor* a = new Tensor(&a_data[0], a_shape);

    float b_data[4] = {6.0f, 2.0f, 4.0f, 3.0f};
    std::vector<size_t> b_shape = {2, 2};

    Tensor* b = new Tensor(&b_data[0], b_shape);

    std::cout << "Checking correctness of Tensor addition and multiplication operations." << std::endl;

    Tensor c = (*a) + (*b);
    print_tensor(c);

    Tensor d = (*a) - (*b);
    print_tensor(d);

    Tensor e = (*b) * 4;
    print_tensor(e);

    Tensor f = (*a) * (*b);
    print_tensor(f);

    Tensor g = (*a) * (*a);
    print_tensor(g);

    Tensor h = (*a) + g;
    print_tensor(h);

    std::cout << "Checking correctness of Tensor gradient operations." << std::endl;

    a->set_requires_grad();
    h.backward();

    print_tensor(*a);
    print_tensor(g);
    print_tensor(h); 

    return 0;
}