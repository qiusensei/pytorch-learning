import torch
import model
import dataset

import matplotlib.pyplot as plt

net = model.MyCNN()
net.load_state_dict(torch.load("../../04_mnist_cnn_v2/checkpoints/mnist_cnn.pth"))
net.eval()

images, labels = next(iter(dataset.test_loader))
print(images[0:1].size())

feature_layer1 = net.net[1](net.net[0](images[0:1]))   # net.net[1] 是 ReLU

plt.figure(figsize=(12,6))                  #创建12x6大小的画布

plt.subplot(4, 9, 1)                        # 第 1 格:原图
plt.imshow(images[0, 0], cmap="gray")
plt.axis("off")

for i in range(feature_layer1.size(1)):
    plt.subplot(4,9,i+2)
    plt.imshow(feature_layer1[0,i].detach(),cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.savefig("../../04_mnist_cnn_v2/plots/feature_layer1_withRAW.png", dpi=150)