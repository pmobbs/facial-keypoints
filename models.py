## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # image size is 224x224
        # kernel size is 5
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # Tensor dimensions will be (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # pooled size will be (32, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)

        # input from first pool = 110x110
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # Tensor dimensions will be (64, 106, 106)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # second pooled size will be (64, 53, 53)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # input from first pool = 53x53
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # Tensor dimensions will be (128, 49, 49)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        # third pooled size will be (128, 24, 24)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # fully-connected layer
        # num_classes outputs (for n_classes of image data)
        self.fc1 = nn.Linear(128*24*24, 136) # was 32*110*110, then 64*53*53
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(136, 136)
        
    # define the feedforward behavior
    def forward(self, x):
        # one conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))

        # two conv/relu + pool layers
        x = self.pool2(F.relu(self.conv2(x)))
        
        # three conv/relu + pool layers
        x = self.pool3(F.relu(self.conv3(x)))

        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        # linear layer 
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        # final output
        return x
