import torch as th
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

x = th.Tensor(3, 2)
# how to create tensor from a list or list of list, workaround is build it from numpy
x = Variable(x, requires_grad=True)
x = Variable(x)  # difference from above
print(x.creator)
y = x * x
print(y.creator)
z = F.tanh(x)
print(z.creator)
z.backward()
print(x.grad)
print(x.data)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        pass

    def forward(self, x):
        # The input to the forward is an autograd.Variable, and so is the output.
        # backward will automatically created for you
        return x


net = Net()
print('NN architecture', net)
print('Learnable parameters', net.parameters())

print(loss.creator)  # MSELoss
print(loss.creator.previous_functions[0][0])  # Linear
print(loss.creator.previous_functions[0][0].previous_functions[0][0])  # ReLU

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)



# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
criterion = nn.MSELoss()
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


z_2 = torch.cat([x_2, y_2], 1)

x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))  # Reshape to 2 rows, 12 columns


# why .grad accumulates value, why its not auto zeroing?

"""
http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
http://pytorch.org/docs/master/nn.html
http://pytorch.org/docs/master/autograd.html
http://pytorch.org/docs/master/torch.html
http://pytorch.org/tutorials/beginner/pytorch_with_examples.html
"""