import os

# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self,args):
        super(CNN, self).__init__()


        self.args = args
        V = args.embed_num
        C = args.class_num
        Co = args.kernel_num
        self.embed = nn.Embedding(V+1, 300, _weight = args.embedding_matrix)
        self.conv1 = nn.Conv2d(
                in_channels=1,              # input height
                out_channels=Co,            # n_filters
                kernel_size=(3,300),              # filter size
                stride=1,                   # filter movement/step
                #padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
        )
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(Co, C)


    def forward(self, x):
        x = self.embed(x) # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = F.relu(self.conv1(x))
        x = x.squeeze(3) #(N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2) #(N, Co)
        x = self.dropout(x)
        output = self.fc(x)
        return output    # return x for visualization