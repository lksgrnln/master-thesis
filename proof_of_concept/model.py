import torch.nn as nn


class MultiLabelClassifier(nn.Module):
    def __init__(self):
        super(MultiLabelClassifier, self).__init__()
        self.output = None
        self.first_convolutional_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(8, 8),
                stride=(4, 4),
                padding=0,
            ),
            nn.ReLU(),
        )
        self.second_convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=(8, 8),
                      stride=(1, 1),
                      padding=0),
            nn.ReLU(),
        )
        # fully connected layer, output 11 classes
        self.first_linear_layer = nn.Linear(32 * 6 * 6, 512)
        self.second_linear_layer = nn.Linear(512, 256)
        self.third_linear_layer = nn.Linear(256, 11)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.first_convolutional_layer(x)
        x = self.second_convolutional_layer(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.first_linear_layer(x))
        x = self.relu(self.second_linear_layer(x))
        self.output = self.third_linear_layer(x)
        return self.output
