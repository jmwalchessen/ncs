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
            nn.BatchNorm2d(out_channels)
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
        self.AvgPoolLayer = nn.AvgPool2d(kernel_size = kernel,
                                         stride = kernel, padding = padding)

    def forward(self, x):
        return self.AvgPoolLayer(x)
    
class DenseLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(DenseLayer, self).__init__()
        self.dense = nn.Sequential(nn.Linear(in_features = in_features, out_features = out_features, bias = True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)
    
class CNNClassifier(nn.Module):

    def __init__(self) -> None:
        super(CNNClassifier, self).__init__()
        cnn_layers = [ConvLayer(in_channels = 1, out_channels = 128, kernel = (3,3)),
                  nn.Tanh(),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 128, out_channels = 128, kernel = (3,3)),
                  nn.Tanh(),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 128, out_channels = 16, kernel = (3,3)),
                  nn.Tanh(),
                  MaxPool2dlayer(kernel = (2,2), padding = (0,0)),
                  nn.Flatten()
                  ]

        dnn_layers = [
                  DenseLayer(in_features = 65, out_features = 32),
                  nn.Tanh(),
                  DenseLayer(in_features = 32, out_features = 16),
                  nn.Tanh(),
                  DenseLayer(in_features = 16, out_features = 8),
                  nn.Tanh(),
                  DenseLayer(in_features = 8, out_features = 1)]
        self.cnn = nn.Sequential(*cnn_layers)
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
        cnn_output = self.cnn(x)
        classifier_output = self.dnn_layers(torch.cat((cnn_output, lengthscale), dim = 1))
        return classifier_output
    
class SmallCNNClassifier(nn.Module):

    def __init__(self) -> None:
        super(SmallCNNClassifier, self).__init__()
        layers = [ConvLayer(in_channels = 1, out_channels = 16, kernel = (3,3)),
                  nn.Tanh(),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 16, out_channels = 8, kernel = (3,3)),
                  nn.Tanh(),
                  MaxPool2dlayer(kernel = (2,2), padding = (1,1)),
                  ConvLayer(in_channels = 8, out_channels = 4, kernel = (3,3)),
                  nn.Tanh(),
                  MaxPool2dlayer(kernel = (2,2), padding = (0,0)),
                  nn.Flatten(),
                  DenseLayer(in_features = 16, out_features = 1)
                  ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


