from tinygrad import Device
from tinygrad import Tensor, nn
from typing import cast
from tinygrad.nn.datasets import mnist

# ttnn_device = Device["TTNN"]

class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)

  def to(self, device:str):
    self.l1.weight = self.l1.weight.to(device)
    if self.l1.bias is not None: self.l1.bias = self.l1.bias.to(device)
    self.l2.weight = self.l2.weight.to(device)
    if self.l2.bias is not None: self.l2.bias = self.l2.bias.to(device)
    self.l3.weight = self.l3.weight.to(device)
    if self.l3.bias is not None: self.l3.bias = self.l3.bias.to(device)
    return self

  def __call__(self, x:Tensor) -> Tensor:
    # x = cast(Tensor, self.l1(x).relu().max_pool2d((2,2)))
    # x = cast(Tensor, self.l2(x).relu().max_pool2d((2,2)))
    x = self.l1(x).relu().max_pool2d((2,2))
    x = self.l2(x).relu().max_pool2d((2,2))
    return self.l3(x.flatten(1).dropout(0.5))
  
X_train, Y_train, X_test, Y_test = mnist(device="TTNN")
print(X_train.shape, X_train.dtype, Y_train.shape, Y_train.dtype)

model = Model().to("TTNN")
acc = (model(X_test).argmax(axis=1) == Y_test).mean()
print(acc.item())