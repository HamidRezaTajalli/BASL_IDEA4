import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, transform=transform, download=True)
testset = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, transform=transform, download=True)


from torch.utils.data import Dataset

class CustomEMNIST(Dataset):
    def __init__(self, emnist_dataset):
        self.dataset = emnist_dataset
        self.indices = [i for i, (_, label) in enumerate(emnist_dataset) if label < 10]

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

trainset = CustomEMNIST(trainset)
testset = CustomEMNIST(testset)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

