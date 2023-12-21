import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from model import MultiLabelClassifier
from dataset import MultiLabelMNIST
from utils import train_ximl
from utils import train
from utils import test
from utils import test_explanations


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiLabelClassifier().to(device)
# model.load_state_dict(torch.load('',
#                                  map_location=torch.device('cpu')
#                                  ))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
model_ximl = MultiLabelClassifier().to(device)
model_ximl.load_state_dict(model.state_dict())
#     '',
#     map_location=torch.device('cpu')
#     ))
optimizer_ximl = torch.optim.Adam(model_ximl.parameters(), lr=1e-3, weight_decay=1e-5)
epochs = 4

transform = transforms.ToTensor()
multilabel_mnist_train_dataset = MultiLabelMNIST(train=True, transform=transform)
multilabel_mnist_test_dataset = MultiLabelMNIST(train=False, transform=transform)
multilabel_mnist_train_loader = DataLoader(multilabel_mnist_train_dataset, batch_size=100, shuffle=True)
multilabel_mnist_test_loader = DataLoader(multilabel_mnist_test_dataset, batch_size=100, shuffle=False)

loss_function = nn.BCEWithLogitsLoss()
train(model, epochs, optimizer, loss_function, multilabel_mnist_train_loader, device)
test(model, multilabel_mnist_test_loader, device, 'common_dl')
train_ximl(model_ximl, epochs, optimizer_ximl, multilabel_mnist_train_loader, device)
test(model_ximl, multilabel_mnist_test_loader, device, 'ximl_dl')
test_explanations(model, multilabel_mnist_test_loader, device)
test_explanations(model_ximl, multilabel_mnist_test_loader, device)
