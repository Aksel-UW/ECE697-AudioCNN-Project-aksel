import dependencies

from torch import nn
from torchinfo import summary

#This code was adapted from the code found at the github page https://github.com/musikalkemist/pytorchforaudio/tree/main/10%20Predictions%20with%20sound%20classifier and through the video tutorials found at https://www.youtube.com/watch?v=gp2wZqDoJ1Y&list=PL-wATfeyAMNoirN4idjev6aRu8ISZYVWm

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 4 conv blocks / flatten / linear / softmax
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 16,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 16,
                out_channels = 32,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 10)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
        
        
if __name__ == "__main__":

    cnn = CNNNetwork()
    summary(cnn.cuda(), (1,1, 64, 44))