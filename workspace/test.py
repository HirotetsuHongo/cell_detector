import torch
import model
import time


# x = torch.randn(6, 1, 416, 416, requires_grad=True).cuda()
# anchors = [[[10, 13], [16, 30], [33, 23]],
#            [[30, 61], [62, 45], [59, 119]],
#            [[116, 90], [156, 198], [373, 326]]]
# f = model.Net(416, 416, 1, 4, anchors).cuda()

x = torch.tensor([3., 2., 4.], requires_grad=True).cuda()

t0 = time.time()
y = x * 10
t1 = time.time()
print("forward time: {:.3f} ms".format((t1-t0)*1000))

t0 = time.time()
y.mean().backward()
t1 = time.time()
print("backward time: {:.3f} ms".format((t1-t0)*1000))

print("x.grad: {}".format(x.grad))
