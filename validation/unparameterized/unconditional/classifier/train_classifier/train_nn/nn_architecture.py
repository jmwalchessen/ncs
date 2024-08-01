import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: tuple) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel, stride = 1, padding = 0),
            #try with and without batch normalization
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x)
        
class MaxPool2dlayer(nn.Module):

    #kernel size is int i.e. assumed to be square (int, int) or tupe (int, int)
    #padding is int or tuple of int
    def __init__(self, kernel: tuple, padding: tuple):
        super(MaxPool2dlayer, self).__init__()
        self.MaxPoolLayer = nn.MaxPool2d(dilation = 1, kernel_size = kernel,
                                         stride = kernel, padding = padding)

    def forward(self, x):
        return self.MaxPoolLayer(x)
    
class AvgPool2dlayer(nn.Module):

    #kernel size is int i.e. assumed to be square (int, int) or tupe (int, int)
    #padding is int or tuple of int
    def __init__(self, kernel: tuple, padding: tuple):
        super(AvgPool2dlayer, self).__init__()
        self.AvgPoolLayer = nn.AvgPool2d(dilation = 1, kernel_size = kernel,
                                         stride = kernel, padding = padding)

    def forward(self, x):
        return self.MaxPoolLayer(x)
    
class DenseLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DenseLayer, self).__init__()
        self.dense = nn.Sequential(nn.Linear(in_features = in_features, out_features = out_features, bias = True), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)
    
class CNNClassifier(nn.Module):

    def __init__(self) -> None:
        super(CNNClassifier, self).__init__()
        layers = [ConvLayer(in_channels = 1, out_channels = 128, kernel = (3,3)),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 128, out_channels = 128, kernel = (3,3)),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 128, out_channels = 16, kernel = (3,3)),
                  MaxPool2dlayer(kernel = (2,2), padding = (0,0)),
                  nn.Flatten(),
                  DenseLayer(in_features = 64, out_features = 32),
                  DenseLayer(in_features = 32, out_features = 16),
                  DenseLayer(in_features = 16, out_features = 8),
                  DenseLayer(in_features = 8, out_features = 2),
                  nn.Sigmoid()
                  ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class SmallCNNClassifier(nn.Module):

    def __init__(self) -> None:
        super(SmallCNNClassifier, self).__init__()
        layers = [ConvLayer(in_channels = 1, out_channels = 16, kernel = (3,3)),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 16, out_channels = 8, kernel = (3,3)),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 8, out_channels = 4, kernel = (3,3)),
                  MaxPool2dlayer(kernel = (2,2), padding = (0,0)),
                  nn.Flatten(),
                  DenseLayer(in_features = 16, out_features = 2),
                  nn.Sigmoid()
                  ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

