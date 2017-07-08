import torch as th
from torch.autograd import Variable

epochs = 501
lr = 1
XOR_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
XOR_Y = [[0, 1], [1, 0], [1, 0], [0, 1]]

if th.cuda.is_available():
    dtype = th.cuda.FloatTensor
else:
    dtype = th.FloatTensor
x = Variable(th.FloatTensor(XOR_X).type(dtype), requires_grad=False)
y = Variable(th.FloatTensor(XOR_Y).type(dtype), requires_grad=False)

w1 = Variable(th.randn(2, 5).type(dtype), requires_grad=True)
w2 = Variable(th.randn(5, 2).type(dtype), requires_grad=True)

b1 = Variable(th.zeros(5).type(dtype), requires_grad=True)
b2 = Variable(th.zeros(2).type(dtype), requires_grad=True)

for epoch in range(epochs):
    a2 = x.mm(w1)
    a2 = a2.add(b1.expand_as(a2))
    h2 = a2.sigmoid()
    a3 = h2.mm(w2)
    a3 = a3.add(b2.expand_as(a3))
    hyp = a3.sigmoid()
    cost = hyp - y
    cost = cost.pow(2).sum()
    if epoch % 500 == 0:
        print(cost.data[0])
    cost.backward()
    w1.data -= lr * w1.grad.data
    w2.data -= lr * w2.grad.data
    b1.data -= lr * b1.grad.data
    b2.data -= lr * b2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()

for x in XOR_X:
    x = Variable(th.FloatTensor([x]))
    a2 = x.mm(w1)
    a2 = a2.add(b1.expand_as(a2))
    h2 = a2.sigmoid()
    a3 = h2.mm(w2)
    a3 = a3.add(b2.expand_as(a3))
    hyp = a3.sigmoid()
    print(x, hyp.max(1)[1].data)
