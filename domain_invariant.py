import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader



import argparse
import pdb


def main():
    parser = argparse.ArgumentParser(description='Domain Agnostic Features')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for real data')

    args=parser.parse_args()
    
    train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5)
    ])
    if args.dataset=='CIFAR10':
       trainset=datasets.CIFAR10(root='./data',train=True,download=True,transform=train_transforms)
        



if __name__=="__main__":
   main()
