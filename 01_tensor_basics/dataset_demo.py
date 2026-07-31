import torch
from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):              #What is the Dataset in the ()?
    def __init__(self,num_samples=100): #What does num_samples mean?
        self.features = torch.randn(num_samples,4)
        self.labels = torch.randint(0,2,(num_samples))


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx],self.labels[idx]

if __name__ == "__main__":
    dataset = ToyDataset()
    print(len(dataset))
    print(dataset[0])

    loader = DataLoader(dataset,batch_size=16,shuffle=True)             #100 can not / 16 and what does shuffle mean?
    print(len(loader))
    for i,(batch_features,batch_labels) in enumerate(loader):           #I forget what is enumerate. And why it has three things i and features and labels?
        print(f"批次{i+1}: features{batch_features.shape},labels{batch_labels.shape}")

'''
所以这里是用DataLoader把原来dataset中的100条sample按照16均分成小batch，多的成为最后一组
shuffle是为了让得到的loader中的sample被打乱
loader本身并不是一个储存数组的块，它是一个用于读取dataset的工具，用这个例子来说，前6个每个读16个，最后一批不够了所以读出来一个4x4
由DataLoader生成的loader本身可以被循环，它很神奇，当你循环它的时候，作为取数器一个个为你取batch，每个batch存在features与batch_labels
所以用 for i ... in enumerate(loader)就可以实现对每个batch挨个编号后挨个操作，比如打印
'''