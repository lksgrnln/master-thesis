import torch.nn as nn


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        if stride == 1:
            self.convolutional_layer_1 = nn.Conv2d(in_channels, out_channels,
                                                   kernel_size=(3, 3), stride=stride, padding=1)
            self.skip_connection = nn.Sequential()
        elif stride == 2:
            self.convolutional_layer_1 = nn.Conv2d(in_channels, out_channels,
                                                   kernel_size=(3, 3), stride=stride, padding=1)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.batchnorm_layer_1 = nn.BatchNorm2d(out_channels)
        self.convolutional_layer_2 = nn.Conv2d(out_channels, out_channels,
                                               kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.batchnorm_layer_2 = nn.BatchNorm2d(out_channels)

    def forward(self, sample):
        skip_connection = self.skip_connection(sample)
        passing_tensor = self.convolutional_layer_1(sample)
        passing_tensor = self.batchnorm_layer_1(passing_tensor)
        passing_tensor = nn.functional.relu(passing_tensor)
        passing_tensor = self.convolutional_layer_2(passing_tensor)
        passing_tensor = self.batchnorm_layer_2(passing_tensor)

        passing_tensor += skip_connection
        output = nn.functional.relu(passing_tensor)

        return output


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.convolutional_layer_1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1))
        self.batchnorm_1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(3, 2)
        self.res_block_1 = ResBlock(64, 64, 1)
        self.res_block_1_2 = ResBlock(64, 64, 1)
        self.res_block_2 = ResBlock(64, 128, 2)
        self.res_block_2_2 = ResBlock(128, 128, 1)
        self.res_block_3 = ResBlock(128, 256, 2)
        self.res_block_3_2 = ResBlock(256, 256, 1)
        self.res_block_4 = ResBlock(256, 512, 2)
        self.res_block_4_2 = ResBlock(512, 512, 1)
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Linear(512, 11)

    def forward(self, sample):
        passing_tensor = self.convolutional_layer_1(sample)
        passing_tensor = self.batchnorm_1(passing_tensor)
        passing_tensor = nn.functional.relu(passing_tensor)
        passing_tensor = self.max_pool(passing_tensor)
        passing_tensor = self.res_block_1(passing_tensor)
        passing_tensor = self.res_block_1_2(passing_tensor)
        passing_tensor = self.res_block_2(passing_tensor)
        passing_tensor = self.res_block_2_2(passing_tensor)
        passing_tensor = self.res_block_3(passing_tensor)
        passing_tensor = self.res_block_3_2(passing_tensor)
        passing_tensor = self.res_block_4(passing_tensor)
        passing_tensor = self.res_block_4_2(passing_tensor)
        passing_tensor = self.global_average_pooling(passing_tensor)
        passing_tensor = self.flatten(passing_tensor)
        passing_tensor = self.fully_connected(passing_tensor)
        output = passing_tensor

        return output
