import copy
import torch
import torch.nn as nn
from rrr_multilabel_loss import rrr_multilabel_loss


def train(_model, epochs, optimizer, loss_function, data_loader):
    num_epochs = epochs
    log = []
    _model.train()

    for i in range(num_epochs):

        running_loss = 0
        for j, data in enumerate(data_loader):

            sample, target, _, _, _, _ = data
            optimizer.zero_grad()

            prediction = _model(sample)
            loss = loss_function(prediction, target)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print('Epoch ' + str(i) + ' Loss: ' + str(running_loss / len(data_loader)))
        log.append('Epoch ' + str(i) + ' Loss: ' + str(running_loss / len(data_loader)) + '\n')
    torch.save(_model.state_dict(), './classifier_common_deep_learning_20_epochs.pth')
    with open("runs/test-run_v5.txt", "w") as text_file:
        for epoch in log:
            text_file.write(epoch)
        text_file.close()


def train_xai(_model, epochs, optimizer, data_loader):
    num_epochs = epochs
    log = []
    _model.train()

    for i in range(num_epochs):

        running_loss = 0
        running_prediction_loss = 0
        for j, data in enumerate(data_loader):

            sample, target_vector, \
                first_target, second_target, \
                first_mask, second_mask = data
            optimizer.zero_grad()

            prediction = _model(sample)
            loss, prediction_loss = rrr_multilabel_loss(_model, sample,
                                                        prediction, target_vector,
                                                        first_target, second_target,
                                                        first_mask, second_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_prediction_loss += prediction_loss.item()

        log.append('Epoch ' + str(i) + ' RRR-Loss: ' + str(running_loss / len(data_loader)) +
                   '; Prediction Loss: ' + str(running_prediction_loss / len(data_loader)) + '\n')
        print('Epoch ' + str(i) + ' RRR-Loss: ' + str(running_loss / len(data_loader)) +
              '; Prediction Loss: ' + str(running_prediction_loss / len(data_loader)))
    torch.save(_model.state_dict(), './classifier_xai_v6_3ep.pth')
    with open("runs/test-run_v6_lr1e-3_3ep_wd1e-5.txt", "w") as text_file:
        for epoch in log:
            text_file.write(epoch)
        text_file.close()


def create_mask(permutation_list, remove_digit):
    background = torch.zeros(1, 28, 28)
    _copy = copy.deepcopy(permutation_list)
    if torch.all(torch.Tensor(_copy[0] == remove_digit)):
        index = 0
    elif torch.all(torch.Tensor(_copy[1] == remove_digit)):
        index = 1
    elif torch.all(torch.Tensor(_copy[2] == remove_digit)):
        index = 2
    else:
        index = 3
    _copy[index] = background
    first_concat = torch.cat((_copy[0], _copy[1]), 2)
    second_concat = torch.cat((_copy[2], _copy[3]), 2)
    mask = torch.cat((first_concat, second_concat), 1)
    inverse = 1 / mask
    inf_indices = inverse == float('inf')
    value_indices = inverse != float('inf')
    inverse[inf_indices] = 1
    inverse[value_indices] = inverse[value_indices] / 255
    inverse[inverse != 1] = 0

    return inverse


def test(model, test_loader):
    both_correct = 0
    single_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            sample, _, target_1, target_2, _, _ = data
            sigmoid = nn.Sigmoid()
            predictions = sigmoid(model(sample))
            predicted_1 = []
            predicted_2 = []
            for prediction in predictions:
                _, predict_1 = torch.max(prediction[0:10], 0)
                _, predict_2 = torch.max(prediction[10:], 0)
                predicted_1.append(predict_1.item())
                predicted_2.append(predict_2.item())
            predicted_1 = torch.Tensor(predicted_1)
            predicted_2 = torch.Tensor(predicted_2)
            total += target_1.size(0)
            both_correct += ((target_1 == predicted_1) & (target_2 == predicted_2)).sum().item()
            single_correct += ((target_1 == predicted_1) | (target_2 == predicted_2)).sum().item()

    print(f'Accuracy of the network on test images for classifying both images correct: '
          f'{100 * both_correct // total} %; accuracy for correct single digit classification: '
          f'{100 * single_correct / (2 * total)} %')
