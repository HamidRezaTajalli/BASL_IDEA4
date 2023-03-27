import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from .trigger import get_backdoor_test_dataset, get_backdoor_train_dataset, GenerateTrigger

import torch
torch.manual_seed(47)
import numpy as np
np.random.seed(47)


class CustomEMNIST(Dataset):
    def __init__(self, emnist_dataset):
        self.dataset = emnist_dataset
        self.indices = [i for i, (_, label) in enumerate(emnist_dataset) if label < 10]

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)



def get_dataloaders_normal(batch_size, drop_last, is_shuffle, trigger_obj, target_label):
    drop_last = drop_last
    is_shuffle = is_shuffle
    batch_size = batch_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=True, transform=transform,
                                           download=True)
    testset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=False, transform=transform,
                                          download=True)

    trainset.targets = trainset.targets - 1
    testset.targets = testset.targets - 1

    classes_names = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]




    train_dataset = CustomEMNIST(trainset)
    test_dataset = CustomEMNIST(testset)
    backdoor_test_dataset = get_backdoor_test_dataset(test_dataset, trigger_obj, trig_ds='emnist',
                                                      backdoor_label=target_label)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                               num_workers=num_workers, drop_last=drop_last)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=is_shuffle,
                                              num_workers=num_workers, drop_last=drop_last)

    backdoor_test_dataloader = torch.utils.data.DataLoader(dataset=backdoor_test_dataset, batch_size=batch_size,
                                                           shuffle=is_shuffle, num_workers=num_workers,
                                                           drop_last=drop_last)

    return {'train': train_dataloader,
            'test': test_dataloader,
            'bd_test': backdoor_test_dataloader}, classes_names

def get_dataloaders_simple(batch_size, train_ds_num, drop_last, is_shuffle):
    drop_last = drop_last
    is_shuffle = is_shuffle
    batch_size = batch_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=True, transform=transform,
                                           download=True)
    testset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=False, transform=transform,
                                          download=True)

    trainset.targets = trainset.targets - 1
    testset.targets = testset.targets - 1

    classes_names = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]




    train_dataset = CustomEMNIST(trainset)
    test_dataset = CustomEMNIST(testset)


    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
                                                                      [int(len(train_dataset) / (12 / 11)),
                                                                       int(len(train_dataset) / (12))])

    chunk_len = len(train_dataset) // train_ds_num
    remnant = len(train_dataset) % train_ds_num
    chunks = [chunk_len for item in range(train_ds_num)]
    if remnant > 0:
        chunks.append(remnant)

    train_datasets = torch.utils.data.random_split(train_dataset, chunks)

    train_dataloaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=batch_size,
                                                     shuffle=is_shuffle, num_workers=num_workers,
                                                     drop_last=drop_last)
                         for i in range(len(train_datasets))]

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)

    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                        shuffle=is_shuffle, num_workers=num_workers,
                                                        drop_last=drop_last)

    return {'train': train_dataloaders,
            'test': test_dataloader,
            'validation': validation_dataloader}, classes_names


def get_dataloaders_lbflip(batch_size, train_ds_num, drop_last, is_shuffle, flip_label, target_label):
    drop_last = drop_last
    batch_size = batch_size
    is_shuffle = is_shuffle
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=True, transform=transform,
                               download=True)
    testset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=False, transform=transform,
                              download=True)

    trainset.targets = trainset.targets - 1
    testset.targets = testset.targets - 1

    classes_names = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    train_dataset = CustomEMNIST(trainset)
    test_dataset = CustomEMNIST(testset)


    target_indices = [num for num, item in enumerate(train_dataset.targets) if item == flip_label]
    rest_indices = [num for num, item in enumerate(train_dataset.targets) if item != flip_label]

    target_dataset = []
    for number, item in enumerate(train_dataset):
        if number in target_indices:
            target_dataset.append((item[0], target_label))
    train_dataset.data = train_dataset.data[rest_indices]
    train_dataset.targets = np.array(train_dataset.targets)[rest_indices].tolist()

    target_indices = [num for num, item in enumerate(test_dataset.targets) if item == flip_label]
    rest_indices = [num for num, item in enumerate(test_dataset.targets) if item != flip_label]

    for number, item in enumerate(test_dataset):
        if number in target_indices:
            target_dataset.append((item[0], target_label))
    test_dataset.data = test_dataset.data[rest_indices]
    test_dataset.targets = np.array(test_dataset.targets)[rest_indices].tolist()

    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
                                                                      [int(len(train_dataset) / (12 / 11)),
                                                                       int(len(train_dataset) / (12))])

    chunk_len = len(train_dataset) // train_ds_num
    remnant = len(train_dataset) % train_ds_num
    chunks = [chunk_len for item in range(train_ds_num)]
    if remnant > 0:
        chunks.append(remnant)
    train_datasets = torch.utils.data.random_split(train_dataset,
                                                   chunks)

    # train_datasets = torch.utils.data.random_split(train_dataset,
    #                                                 [int(len(train_dataset) / (8 / 1)),
    #                                                 int(len(train_dataset) / (8 / 7))])

    backdoor_train_dataset = [item for item in train_datasets[0]]
    backdoor_train_dataset.extend(target_dataset)

    train_datasets[0] = backdoor_train_dataset

    train_dataloaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=batch_size,
                                                     shuffle=is_shuffle, num_workers=num_workers,
                                                     drop_last=drop_last)
                         for i in range(len(train_datasets))]

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers,
                                                  drop_last=drop_last)

    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                        shuffle=is_shuffle, num_workers=num_workers,
                                                        drop_last=drop_last)

    backdoor_test_dataloader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=batch_size,
                                                           shuffle=is_shuffle, num_workers=num_workers,
                                                           drop_last=drop_last)

    #
    # for item in splitted_train_dataset[0]:
    #     if int(item[1]) == 9:
    #         insert = (item[0], 0)
    #         outliers_ds.append(insert)
    #         poison_ds.append(insert)
    #     else:
    #         clean_ds_train.append(item)
    #
    # for item in splitted_train_dataset[1]:
    #     if int(item[1]) == 9:
    #         insert = (item[0], 0)
    #         outliers_ds.append(insert)
    #         poison_ds.append(insert)
    #     else:
    #         poison_ds.append(item)
    #
    # for item in test_dataset:
    #     if int(item[1]) == 9:
    #         insert = (item[0], 0)
    #         outliers_ds.append(insert)
    #         poison_ds.append(insert)
    #     else:
    #         clean_ds_test.append(item)
    #
    # train_dataloader = torch.utils.data.DataLoader(dataset=clean_ds_train, batch_size=batch_size,
    #                                                shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    # test_dataloader = torch.utils.data.DataLoader(dataset=clean_ds_test, batch_size=batch_size,
    #                                               shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    # poison_dataloader = torch.utils.data.DataLoader(dataset=poison_ds, batch_size=batch_size,
    #                                                 shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)
    # outlier_dataloader = torch.utils.data.DataLoader(dataset=outliers_ds, batch_size=batch_size,
    #                                                  shuffle=is_shuffle, num_workers=num_workers, drop_last=drop_last)

    return {'train': train_dataloaders,
            'validation': validation_dataloader,
            'test': test_dataloader,
            'backdoor_test': backdoor_test_dataloader}, classes_names


def get_dataloaders_backdoor(batch_size, train_ds_num, drop_last, is_shuffle, target_label, train_samples_percentage):
    drop_last = drop_last
    batch_size = batch_size
    is_shuffle = is_shuffle
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 2 if device.type == 'cuda' else 0

    transform = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=True, transform=transform,
                               download=True)
    testset = datasets.EMNIST(root='./data/EMNIST/', split='letters', train=False, transform=transform,
                              download=True)

    trainset.targets = trainset.targets - 1
    testset.targets = testset.targets - 1

    classes_names = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    train_dataset = CustomEMNIST(trainset)
    test_dataset = CustomEMNIST(testset)


    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset,
                                                                      [int(len(train_dataset) / (12 / 11)),
                                                                       int(len(train_dataset) / (12))])

    chunk_len = len(train_dataset) // train_ds_num
    remnant = len(train_dataset) % train_ds_num
    chunks = [chunk_len for item in range(train_ds_num)]
    if remnant > 0:
        chunks.append(remnant)
    train_datasets = torch.utils.data.random_split(train_dataset,
                                                   chunks)
    # train_datasets = torch.utils.data.random_split(train_dataset,
    #                                                [int(len(train_dataset) / (8 / 1)),
    #                                                 int(len(train_dataset) / (8 / 7))])

    trigger_obj = GenerateTrigger((8, 8), pos_label='upper-mid', dataset='emnist', shape='square')

    train_datasets[0] = get_backdoor_train_dataset(train_datasets[0], trigger_obj, trig_ds='emnist',
                                                   samples_percentage=train_samples_percentage,
                                                   backdoor_label=target_label)

    backdoor_test_dataset = get_backdoor_test_dataset(test_dataset, trigger_obj, trig_ds='emnist',
                                                      backdoor_label=target_label)
    train_dataloaders = [torch.utils.data.DataLoader(dataset=train_datasets[i], batch_size=batch_size,
                                                     shuffle=is_shuffle, num_workers=num_workers,
                                                     drop_last=drop_last)
                         for i in range(len(train_datasets))]

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                  shuffle=is_shuffle, num_workers=num_workers,
                                                  drop_last=drop_last)

    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                        shuffle=is_shuffle, num_workers=num_workers,
                                                        drop_last=drop_last)

    backdoor_test_dataloader = torch.utils.data.DataLoader(dataset=backdoor_test_dataset, batch_size=batch_size,
                                                           shuffle=is_shuffle, num_workers=num_workers,
                                                           drop_last=drop_last)

    return {'train': train_dataloaders,
            'validation': validation_dataloader,
            'test': test_dataloader,
            'backdoor_test': backdoor_test_dataloader}, classes_names
