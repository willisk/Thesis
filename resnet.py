
net = ResNet(10, [3, 3, 3])
inputs = torch.rand((8, 3, 32, 32))

print(net(inputs))
