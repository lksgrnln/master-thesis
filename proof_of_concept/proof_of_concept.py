import torch
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from model import MultiLabelClassifier
from dataset import MultiLabelMNIST
from utils import train_xai
from utils import train
from utils import test


# torch.manual_seed(64)
# random.seed(64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiLabelClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model_xai = MultiLabelClassifier().to(device)
model_xai.load_state_dict(model.state_dict())
optimizer_xai = torch.optim.Adam(model_xai.parameters(), lr=1e-3, weight_decay=1e-5)
epochs = 200

transform = transforms.ToTensor()
multilabel_mnist_train_dataset = MultiLabelMNIST(train=True, transform=transform)
multilabel_mnist_test_dataset = MultiLabelMNIST(train=False, transform=transform)
multilabel_mnist_train_loader = DataLoader(multilabel_mnist_train_dataset, batch_size=100, shuffle=True)
multilabel_mnist_test_loader = DataLoader(multilabel_mnist_test_dataset, batch_size=100, shuffle=True)

loss_function = nn.BCEWithLogitsLoss()
train(model, epochs, optimizer, loss_function, multilabel_mnist_train_loader, device)
test(model, multilabel_mnist_test_loader, device)

train_xai(model_xai, epochs, optimizer_xai, multilabel_mnist_train_loader, device)
test(model_xai, multilabel_mnist_test_loader, device)
# test_images, labels, first_target, second_target, first_mask, second_mask = next(iter(multilabel_mnist_test_loader))
# print(first_target[1])
# print(second_target[1])
# plt.imshow(test_images[0].reshape(56, 56), cmap='gray')
# plt.show()
# plt.imshow(first_mask[0].reshape(56, 56), cmap='gray')
# plt.show()
# plt.imshow(second_mask[0].reshape(56, 56), cmap='gray')
# plt.show()
# sigmoid = nn.Sigmoid()
# prediction = sigmoid(model(test_images[1].reshape(1,1,56,56)))
# print(prediction)

# integrated_gradient = IntegratedGradients(model)
# print(test_images[0].unsqueeze(0).shape)
# attribution1 = integrated_gradient.attribute(test_images[1].unsqueeze(0), target=2)
# torch.set_printoptions(threshold=10000)
# attribution1[attribution1 < 0] = 0
# print(attribution)
# plt.imshow(attribution1.reshape(56, 56), cmap='gray')
# plt.show()
# attribution2 = integrated_gradient.attribute(test_images[1].unsqueeze(0), target=11)
# attribution2[attribution2 < 0] = 0
# plt.imshow(attribution2.reshape(56, 56), cmap='gray')
# plt.show()
