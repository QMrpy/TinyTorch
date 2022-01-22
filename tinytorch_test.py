import tinytorch_py as torch

a = torch.Tensor([1, 2, 3], 3)
b = torch.Tensor([3, 4, 5], 3)

print("a = ", a)
print("b = ", b)
print("a.size() = ", a.size())
print("b.size() = ", b.size())

print(a + b)