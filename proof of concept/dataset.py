import torch
import random
from utils import create_mask
from torchvision import datasets
from torch.utils.data import Dataset


class MultiLabelMNIST(Dataset):

    def __init__(self, train, transform):
        self.train = train
        self.transform = transform
        self.mnist_train = datasets.MNIST(root='.', train=True, download=False, transform=transform)
        self.mnist_test = datasets.MNIST(root='.', train=False, download=False, transform=transform)

    def __len__(self):
        if self.train:
            return len(self.mnist_train)
        else:
            return len(self.mnist_test)

    def __getitem__(self, index):
        if self.train:
            if index == 59999:
                index = 59998
            first_digit, first_target = self.mnist_train[index][0], self.mnist_train[index][1]
            second_digit, second_target = self.mnist_train[index + 1][0], self.mnist_train[index + 1][1]
            target = torch.zeros(20)
            target[first_target] = 1
            target[second_target + 10] = 1
            background = torch.zeros(1, 28, 28)
            permutation_list = [first_digit, second_digit,
                                background, background]
            random.shuffle(permutation_list)
            first_concat = torch.cat((permutation_list[0], permutation_list[1]), 2)
            second_concat = torch.cat((permutation_list[2], permutation_list[3]), 2)
            sample = torch.cat((first_concat, second_concat), 1)
            first_mask = create_mask(permutation_list, second_digit)
            second_mask = create_mask(permutation_list, first_digit)

        else:
            if index == 9999:
                index = 9998
            first_digit, first_target = self.mnist_test[index][0], self.mnist_test[index][1]
            second_digit, second_target = self.mnist_test[index + 1][0], self.mnist_test[index + 1][1]
            target = torch.zeros(20)
            target[first_target] = 1
            target[second_target + 10] = 1
            background = torch.zeros(1, 28, 28)
            permutation_list = [first_digit, second_digit,
                                background, background]
            random.shuffle(permutation_list)
            first_concat = torch.cat((permutation_list[0], permutation_list[1]), 2)
            second_concat = torch.cat((permutation_list[2], permutation_list[3]), 2)
            sample = torch.cat((first_concat, second_concat), 1)
            first_mask = create_mask(permutation_list, second_digit)
            second_mask = create_mask(permutation_list, first_digit)

        return sample, target, first_target, second_target, first_mask, second_mask
