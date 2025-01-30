import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#loading dataset
#three way split: train-valid-test
def get_train_valid_loader(
        data_dir,
        batch_size,
        augment,
        random_seed,
        valid_size = 0.1,
        shuffle = True
): 
    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2020],
    ) #normalization values calculated for CIFAR10 datasets

    valid_transform = transforms.Compose([
        transforms.Resize((227,227)), #resize
        transforms.ToTensor(), #convert to tensor
        normalize, #pixel normalization
    ]) #validation datast transform is only resized and normalized

    if augment:

        train_transform = transforms.Compose([ #combines multiple transforms
            transforms.Resize((227,227)),  #resize to 227 * 227
            transforms.RandomCrop(227, padding = 4), # add padding and randomly crop 
            transforms.RandomHorizontalFlip(), #random horizontal flip
            transforms.ToTensor(), #convert to tensor
            normalize #pixel normalization

        ])

    train_dataset = datasets.CIFAR10(
        root = data_dir, train = True,
        download = True, transform = train_transform,
    ) #training dataset transform

    valid_dataset = datasets.CIFAR10(
        root = data_dir, train = True,
        download= True, transform = valid_transform
    ) #validation dataset transform

    num_train = len(train_dataset)  #total number of samples
    indices = list(range(num_train)) #create list 
    split = int(np.floor(valid_size * num_train)) #find validation set size

    if shuffle:

        np.random.seed(random_seed) 
        np.random.shuffle(indices) 


    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size, train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size, 
    )

    return (train_loader, valid_loader )


def get_test_loader(data_dir, batch_size, shuffle = True) :
    normalize = transforms.Normalize(

        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]

    )

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
         normalize,
    ])

    dataset = datasets.CIFAR10(
        root = data_dir, train = False, #separate test dataset
        download = True, transform = transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle
    )

    return data_loader