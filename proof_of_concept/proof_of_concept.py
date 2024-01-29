import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from utils import test
from utils import train
from pathlib import Path
from resnet18 import ResNet
from utils import train_ximl
from torchvision import transforms
from utils import test_explanations
from dataset import MultiLabelMNIST
from model import MultiLabelClassifier
from torch.utils.data import DataLoader
from utils import mean_and_standard_deviation


torch.manual_seed(64)
random.seed(64)
np.random.seed(64)

if not Path("./models").is_dir():
    Path("./models").mkdir()
if not Path("./runs").is_dir():
    Path("./runs").mkdir()

parser = argparse.ArgumentParser(description="XIML execution script.")
parser.add_argument(
        "-m",
        "--model",
        default="MLC",
        help="Model architecture (default: %(default)s)",
    )
parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
)
parser.add_argument(
        "-e_ximl",
        "--epochs_ximl",
        default=10,
        type=int,
        help="Number of ximl epochs (default: %(default)s)",
)
parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Learning rate (default: %(default)s)",
)
parser.add_argument(
        "-lr_ximl",
        "--learning_rate_ximl",
        type=float,
        default=1e-3,
        help="XIML learning rate (default: %(default)s)",
)
parser.add_argument(
    "-bs",
    "--batch-size",
    type=int,
    default=100,
    help="Batch size (default: %(default)s)"
)
parser.add_argument(
    "-wd",
    "--weight_decay",
    type=int,
    default=1e-5,
    help="Weight decay (default: %(default)s)"
)
parser.add_argument(
    "-seq",
    "--sequential",
    default=False,
    action="store_true",
    help="Sequential processing with XIML fine-tuning (default: %(default)s)"
)
parser.add_argument(
    "-m_std",
    "--m_std",
    default=False,
    action="store_true",
    help="Compute mean and standard deviation over model ensemble (default: %(default)s)"
)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if args.model == "MLC":
    model = MultiLabelClassifier().to(device)
else:
    model = ResNet().to(device)
# model.load_state_dict(torch.load('./models/classifier_common_deep_learning_11_20231222185557.pth',
#                                  map_location=torch.device('cpu')
#                                  ))
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay)
if args.model == "MLC":
    model_ximl = MultiLabelClassifier().to(device)
else:
    model_ximl = ResNet().to(device)
model_ximl.load_state_dict(model.state_dict())
# model_ximl.load_state_dict(torch.load('./models/classifier_ximl_output_11_20231226151411.pth',
#                                       map_location=torch.device('cpu')
#                                       ))
optimizer_ximl = torch.optim.Adam(model.parameters(),
                                  lr=args.learning_rate_ximl,
                                  weight_decay=args.weight_decay)
epochs = args.epochs
epochs_ximl = args.epochs_ximl

transform = transforms.ToTensor()
multilabel_mnist_train_dataset = MultiLabelMNIST(train=True, transform=transform)
multilabel_mnist_test_dataset = MultiLabelMNIST(train=False, transform=transform)
multilabel_mnist_train_loader = DataLoader(multilabel_mnist_train_dataset,
                                           batch_size=args.batch_size, shuffle=True)
multilabel_mnist_test_loader = DataLoader(multilabel_mnist_test_dataset,
                                          batch_size=args.batch_size, shuffle=False)
loss_function = nn.BCEWithLogitsLoss()

if args.sequential:
    train(model, epochs, optimizer, loss_function, multilabel_mnist_train_loader, device)
    test(model, multilabel_mnist_test_loader, device, 'common_dl')
    test_explanations(model, multilabel_mnist_test_loader,
                      device, 'number of high activated pixels', 'dl')
    train_ximl(model, epochs_ximl, optimizer_ximl,
               multilabel_mnist_train_loader, device, args.model)
    test(model, multilabel_mnist_test_loader, device, 'ximl_dl')
    test_explanations(model, multilabel_mnist_test_loader,
                      device, 'number of high activated pixels', 'ximl')
else:
    train(model, epochs, optimizer, loss_function, multilabel_mnist_train_loader, device)
    test(model, multilabel_mnist_test_loader, device, 'common_dl')
    test_explanations(model, multilabel_mnist_test_loader,
                      device, 'number of high activated pixels', 'dl')
    train_ximl(model_ximl, epochs_ximl, optimizer_ximl,
               multilabel_mnist_train_loader, device, args.model)
    test(model_ximl, multilabel_mnist_test_loader, device, 'ximl_dl')
    test_explanations(model_ximl, multilabel_mnist_test_loader,
                      device, 'number of high activated pixels', 'ximl')

if args.m_std:
    mean_and_standard_deviation()
