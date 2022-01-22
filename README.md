# TinyTorch
A basic AutoGrad Engine written from scratch in C++. Constructs a computation graph out of operations on Tensors, computes and stores gradients in the nodes, similar to PyTorch. The API is PyTorch like.

For now, only addition, subtraction, scalar multiplication and element-wise tensor multiplication are supported, but it is easily extensible to any arbitrary differentiable operation. A separate module for such operations, similar to PyTorch's `nn.Module` is planned.

Python3 bindings are also provided, however, the binding code doesn't work properly yet as the C++ code uses C style float arrays, which PyBind11 can't handle as it doesn't have pointers. The constructor function argument types may need to be changed, and/or the underlying data type of the Tensor data and gradient from raw C-style arrays to C++ vectors.

## Steps to build and run the C++ code (Mac OSX Big Sur)

1. Ensure that CMake and C++ >= 17 are installed.
2. Clone the repository.
3. Perform,
    ```
    cd src
    mkdir build
    cd build
    cmake ..
    make
    ./tinytorch
    ```
   This will run the code and test all the operations. Both data and gradients of the outputs can be matched with PyTorch to check for correctness.

## Steps to build the Python bindings from C++ (Mac OSX Big Sur)

1. Install PyBind11,
    ```
    brew install pybind11
    ```
2. Perform,
    ```
    cd python
    mkdir build
    cd build
    cmake .. -DPYTHON_LIBRARY_DIR="/usr/local/lib/python3.x/site-packages" -DPYTHON_EXECUTABLE=`which python3`
    make
    make install
    cd ../..
    python3 tinytorch_test.py
    ```
   This should output an error such as this, 
    ```
        a = torch.Tensor([1, 2, 3], 3)
    TypeError: __init__(): incompatible constructor arguments. The following argument types are supported:
        1. tinytorch_py.Tensor(arg0: int)
        2. tinytorch_py.Tensor(arg0: List[int])
        3. tinytorch_py.Tensor(arg0: float, arg1: int)
        4. tinytorch_py.Tensor(arg0: float, arg1: int, arg2: bool)
        5. tinytorch_py.Tensor(arg0: float, arg1: List[int])
        6. tinytorch_py.Tensor(arg0: float, arg1: List[int], arg2: bool)

    Invoked with: [1, 2, 3], 3
    ```
   Anything else means that the python module is not properly working. It may happen that the system can't find the appropriate `.so` file from the appropriate python `site-packages` directory. In that case, copy the `.so` file generated in the path which appears in the `make` logs to the `site-packages` directory which is being used by the system. 

