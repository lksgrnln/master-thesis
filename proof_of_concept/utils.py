import os
import copy
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as func
from datetime import datetime as dt
from captum.attr import IntegratedGradients
from rrr_multilabel_loss import rrr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score


def train(_model, epochs, optimizer, loss_function, data_loader, device):
    num_epochs = epochs
    log = []
    training_loss_per_epoch = {}
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    _model.train()

    for i in range(num_epochs):

        running_loss = 0
        for j, data in enumerate(data_loader):

            sample, target, _, _, _, _ = data
            optimizer.zero_grad()

            prediction = _model(sample.to(device))
            loss = loss_function(prediction, target.to(device))
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        scheduler.step(running_loss / len(data_loader))
        training_loss_per_epoch[i] = (running_loss / len(data_loader))
        print('Epoch ' + str(i) + ' Loss: ' + str(running_loss / len(data_loader)) +
              '; Learning rate: ' + str(optimizer.defaults['lr']))
        log.append('Epoch ' + str(i) + ' Loss: ' + str(running_loss / len(data_loader)) + '\n')
    torch.save(_model.state_dict(), './models/classifier_common_deep_learning_11_' +
               dt.strftime(dt.now(), "%Y%m%d%H%M%S") + '.pth')
    with open(
            "./runs/training_loss_common_dl_" +
            dt.strftime(dt.now(), "%Y%m%d%H%M%S") +
            ".json", "w"
    ) as outfile:
        json.dump(training_loss_per_epoch, outfile, indent=2)
    print('finished common training')


def train_ximl(_model, epochs, optimizer, data_loader, device, model_type, ximl_lr):
    num_epochs = epochs
    log = []
    training_loss_per_epoch = {}
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10)
    _model.train()

    for i in range(num_epochs):

        running_loss = 0
        running_prediction_loss = 0
        running_explanation_loss = 0
        for j, data in enumerate(data_loader):

            sample, target_vector, \
                first_target, second_target, \
                first_mask, second_mask = data
            optimizer.zero_grad()

            prediction = _model(sample.to(device))
#             loss, prediction_loss, explanation_loss = rrr_multilabel_loss(
#                 _model, sample.to(device),
#                 prediction.to(device), target_vector.to(device),
#                 first_target.to(device), second_target.to(device),
#                 first_mask.to(device), second_mask.to(device),
#                 device
#             )
            loss, prediction_loss, explanation_loss = rrr(
                _model, sample.to(device),
                prediction.to(device), target_vector.to(device),
                first_target.to(device), second_target.to(device),
                first_mask.to(device), second_mask.to(device),
                device, model_type, ximl_lr
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_prediction_loss += prediction_loss.item()
            running_explanation_loss += explanation_loss.item()
        scheduler.step(running_loss / len(data_loader))
        training_loss_per_epoch[i] = running_loss / len(data_loader), \
            running_prediction_loss / len(data_loader), \
            running_explanation_loss / len(data_loader)
        log.append('Epoch ' + str(i) + ' RRR-Loss: ' + str(running_loss / len(data_loader)) +
                   '; Prediction Loss: ' + str(running_prediction_loss / len(data_loader)) + '\n')
        print('Epoch ' + str(i) + ' RRR-Loss: ' + str(running_loss / len(data_loader)) +
              '; Prediction Loss: ' + str(running_prediction_loss / len(data_loader)) +
              '; Learning rate: ' + str(optimizer.defaults['lr']))
    torch.save(_model.state_dict(), './models/classifier_ximl_output_11_' +
               dt.strftime(dt.now(), "%Y%m%d%H%M%S") + '.pth')
    with open(
            "./runs/training_loss_ximl_dl_" +
            dt.strftime(dt.now(), "%Y%m%d%H%M%S") +
            ".json", "w") as outfile:
        json.dump(training_loss_per_epoch, outfile, indent=2)
    print('finished ximl training')


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


def test(model, test_loader, device, report_name):
    both_correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        test_set_targets = torch.zeros(len(test_loader) * len(next(iter(test_loader))[0]), 11)
        test_set_predictions = torch.zeros(len(test_loader) * len(next(iter(test_loader))[0]), 11)
        for i, data in enumerate(test_loader):
            sample, target_vector, target_1, target_2, _, _ = data
            test_set_targets[i*len(sample): i*len(sample) + len(sample)] = target_vector
            sigmoid = nn.Sigmoid().to(device)
            target_1 = target_1.to(device)
            target_2 = target_2.to(device)
            _, predictions = torch.topk(sigmoid(model(sample.to(device))), 2)
            prediction_vector = torch.sum(torch.nn.functional.one_hot(predictions), dim=1)
            if prediction_vector.shape[1] < 11:
                pad_zeros = 11 - prediction_vector.shape[1]
                prediction_vector = torch.nn.functional.pad(prediction_vector, (0, pad_zeros), "constant", 0)
            test_set_predictions[i*len(sample): i*len(sample) + len(sample)] = prediction_vector
            predicted_1 = predictions[:, 0].to(device)
            predicted_2 = predictions[:, 1].to(device)
            predicted_1[predicted_1 == 10] = predicted_2[predicted_1 == 10]
            predicted_2[predicted_2 == 10] = predicted_1[predicted_2 == 10]
            total += target_1.size(0)
            both_correct += (((predicted_1 == target_1) | (predicted_1 == target_2)).sum().item() +
                             ((predicted_2 == target_1) | (predicted_2 == target_2)).sum().item()) / 2

    report = classification_report(test_set_targets, test_set_predictions, output_dict=True)
    report["accuracy_score"] = accuracy_score(test_set_targets, test_set_predictions)
    with open(
            "./runs/report_" + report_name + "_" +
            dt.strftime(dt.now(), "%Y%m%d%H%M%S") +
            ".json", "w") as outfile:
        json.dump(report, outfile, indent=2)
    print('sklearn accuracy: ' + str(accuracy_score(test_set_targets, test_set_predictions)))
    print(f'Accuracy of the network on test images for classifying both images correct: '
          f'{100 * both_correct / total} %')


def test_explanations(model, test_loader, device, mode, test_score):
    if mode == 'cosine_similarity':
        model.eval()
        with torch.no_grad():
            running_similarity = 0
            for data in test_loader:
                sample, _, target_1, target_2, mask_1, mask_2 = data
                sample = sample.to(device)
                target_1 = target_1.to(device)
                target_2 = target_2.to(device)
                mask_1 = mask_1.to(device)
                mask_2 = mask_2.to(device)
                explanation_target_1 = torch.zeros_like(mask_1).to(device)
                explanation_target_2 = torch.zeros_like(mask_2).to(device)
                index0_1 = mask_1 == 0
                index1_1 = mask_1 == 1
                index0_2 = mask_2 == 0
                index1_2 = mask_2 == 1
                explanation_target_1[index0_1] = 1
                explanation_target_1[index1_1] = 0
                explanation_target_2[index0_2] = 1
                explanation_target_2[index1_2] = 0
                explanation_target_1 = torch.flatten(explanation_target_1)
                explanation_target_2 = torch.flatten(explanation_target_2)
                integrated_gradient = IntegratedGradients(model)
                gradient_tensor_1 = torch.flatten(integrated_gradient.attribute(sample, target=target_1).to(device))
                gradient_tensor_1[gradient_tensor_1 < 0] = 0
                gradient_tensor_1 /= gradient_tensor_1.max()
                gradient_tensor_1[gradient_tensor_1 > 0] = 1
                gradient_tensor_2 = torch.flatten(integrated_gradient.attribute(sample, target=target_2).to(device))
                gradient_tensor_2[gradient_tensor_2 < 0] = 0
                gradient_tensor_2 /= gradient_tensor_2.max()
                gradient_tensor_2[gradient_tensor_2 > 0] = 1
                cosine_similarity_1 = func.cosine_similarity(explanation_target_1, gradient_tensor_1, dim=0)
                cosine_similarity_2 = func.cosine_similarity(explanation_target_2, gradient_tensor_2, dim=0)
                running_similarity += (cosine_similarity_1 + cosine_similarity_2) / 2
            similarity = running_similarity / len(test_loader)
            print(f'cosine similarity of explanations: {similarity.item()}')
    else:
        model.eval()
        with torch.no_grad():
            count_right_high_activated_pixels = 0
            count_wrong_high_activated_pixels = 0
            for data in test_loader:
                sample, _, target_1, target_2, mask_1, mask_2 = data
                sample = sample.to(device)
                target_1 = target_1.to(device)
                target_2 = target_2.to(device)
                mask_1 = mask_1.to(device)
                mask_2 = mask_2.to(device)
                integrated_gradient = IntegratedGradients(model)
                gradient_tensor_1 = integrated_gradient.attribute(sample, target=target_1).to(device)
                gradient_tensor_1[gradient_tensor_1 < 0] = 0
                for explanation in gradient_tensor_1:
                    explanation[explanation < (explanation.max() / 3)] = 0
                gradient_tensor_1[gradient_tensor_1 > 0] = 1
                count_right_high_activated_pixels += (gradient_tensor_1 * mask_2).sum()
                count_wrong_high_activated_pixels += (gradient_tensor_1 * mask_1).sum()
                gradient_tensor_2 = integrated_gradient.attribute(sample, target=target_2).to(device)
                gradient_tensor_2[gradient_tensor_2 < 0] = 0
                for explanation in gradient_tensor_2:
                    explanation[explanation < (explanation.max() / 3)] = 0
                gradient_tensor_2[gradient_tensor_2 > 0] = 1
                count_right_high_activated_pixels += (gradient_tensor_2 * mask_1).sum()
                count_wrong_high_activated_pixels += (gradient_tensor_2 * mask_2).sum()
            print('number of high activated right pixels: ' + str(count_right_high_activated_pixels.item()))
            print('number of high activated wrong pixels: ' + str(count_wrong_high_activated_pixels.item()))
            print('explanation score of classifier: ' + str(count_right_high_activated_pixels.item() /
                                                            count_wrong_high_activated_pixels.item()))
            score = {
                "high_activated_right_pixels": count_right_high_activated_pixels.item(),
                "high_activated_wrong_pixels": count_wrong_high_activated_pixels.item(),
                "explanation_score": (count_right_high_activated_pixels.item() /
                                      count_wrong_high_activated_pixels.item())
            }
            with open(
                    "./runs/multi_label_explanation_score" + "_" +
                    str(test_score) + "_" +
                    dt.strftime(dt.now(), "%Y%m%d%H%M%S") +
                    ".json", "w") as outfile:
                json.dump(score, outfile, indent=2)


def mean_and_standard_deviation():
    report_dict_dl = {
        "0": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "1": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "2": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "3": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "4": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "5": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "6": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "7": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "8": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "9": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "10": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "macro avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "weighted avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "samples avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "accuracy_score": 0
    }
    report_dict_dl_std = {
        "0": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "1": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "2": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "3": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "4": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "5": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "6": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "7": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "8": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "9": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "10": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "macro avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "weighted avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "samples avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "accuracy_score": []
    }
    report_dict_ximl = {
        "0": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "1": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "2": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "3": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "4": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "5": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "6": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "7": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "8": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "9": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "10": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "macro avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "weighted avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "samples avg": {
            "precision": 0,
            "recall": 0,
            "f1-score": 0,
            "support": 0
        },
        "accuracy_score": 0
    }
    report_dict_ximl_std = {
        "0": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "1": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "2": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "3": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "4": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "5": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "6": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "7": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "8": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "9": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "10": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "macro avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "weighted avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "samples avg": {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": []
        },
        "accuracy_score": []
    }
    explanation_score_dict_dl_std = {}
    explanation_score_dict_ximl_std = {}
    explanation_score_dict_dl_mean = {
        "high_activated_right_pixels": 0,
        "high_activated_wrong_pixels": 0,
        "explanation_score": 0
    }
    explanation_score_dict_ximl_mean = {
        "high_activated_right_pixels": 0,
        "high_activated_wrong_pixels": 0,
        "explanation_score": 0
    }
    file_list = []

    for i, (root, dirs, files) in enumerate(os.walk("./runs")):
        if i == 0:
            for file in files:
                file_list.append(file)
        else:
            continue
    count_dl = 0
    count_ximl = 0
    count_r_dl = 0
    count_r_ximl = 0
    score_dl_std_list = [[], [], []]
    score_ximl_std_list = [[], [], []]
    for json_file in file_list:
        if "score_dl" in json_file:
            with open("./runs/" + json_file) as json_data:
                d = json.load(json_data)
                explanation_score_dict_dl_mean[
                    "high_activated_right_pixels"
                ] += d["high_activated_right_pixels"]
                score_dl_std_list[0].append(d["high_activated_right_pixels"])
                explanation_score_dict_dl_mean[
                    "high_activated_wrong_pixels"
                ] += d["high_activated_wrong_pixels"]
                score_dl_std_list[1].append(d["high_activated_wrong_pixels"])
                explanation_score_dict_dl_mean[
                    "explanation_score"
                ] += d["explanation_score"]
                score_dl_std_list[2].append(d["explanation_score"])
                count_dl += 1
        elif "score_ximl" in json_file:
            with open("./runs/" + json_file) as json_data:
                d = json.load(json_data)
                explanation_score_dict_ximl_mean[
                    "high_activated_right_pixels"
                ] += d["high_activated_right_pixels"]
                score_ximl_std_list[0].append(d["high_activated_right_pixels"])
                explanation_score_dict_ximl_mean[
                    "high_activated_wrong_pixels"
                ] += d["high_activated_wrong_pixels"]
                score_ximl_std_list[1].append(d["high_activated_wrong_pixels"])
                explanation_score_dict_ximl_mean[
                    "explanation_score"
                ] += d["explanation_score"]
                score_ximl_std_list[2].append(d["explanation_score"])
                count_ximl += 1
        elif "report_common" in json_file:
            with open("./runs/" + json_file) as json_data:
                d = json.load(json_data)
                for i in range(11):
                    report_dict_dl[str(i)]["precision"] += d[str(i)]["precision"]
                    report_dict_dl_std[str(i)]["precision"].append(d[str(i)]["precision"])
                    report_dict_dl[str(i)]["recall"] += d[str(i)]["recall"]
                    report_dict_dl_std[str(i)]["recall"].append(d[str(i)]["recall"])
                    report_dict_dl[str(i)]["f1-score"] += d[str(i)]["f1-score"]
                    report_dict_dl_std[str(i)]["f1-score"].append(d[str(i)]["f1-score"])
                    report_dict_dl[str(i)]["support"] += d[str(i)]["support"]
                    report_dict_dl_std[str(i)]["support"].append(d[str(i)]["support"])
                report_dict_dl["macro avg"]["precision"] += d["macro avg"]["precision"]
                report_dict_dl_std["macro avg"]["precision"].append(d["macro avg"]["precision"])
                report_dict_dl["macro avg"]["recall"] += d["macro avg"]["recall"]
                report_dict_dl_std["macro avg"]["recall"].append(d["macro avg"]["recall"])
                report_dict_dl["macro avg"]["f1-score"] += d["macro avg"]["f1-score"]
                report_dict_dl_std["macro avg"]["f1-score"].append(d["macro avg"]["f1-score"])
                report_dict_dl["macro avg"]["support"] += d["macro avg"]["support"]
                report_dict_dl_std["macro avg"]["support"].append(d["macro avg"]["support"])
                report_dict_dl["weighted avg"]["precision"] += d["weighted avg"]["precision"]
                report_dict_dl_std["weighted avg"]["precision"].append(d["weighted avg"]["precision"])
                report_dict_dl["weighted avg"]["recall"] += d["weighted avg"]["recall"]
                report_dict_dl_std["weighted avg"]["recall"].append(d["weighted avg"]["recall"])
                report_dict_dl["weighted avg"]["f1-score"] += d["weighted avg"]["f1-score"]
                report_dict_dl_std["weighted avg"]["f1-score"].append(d["weighted avg"]["f1-score"])
                report_dict_dl["weighted avg"]["support"] += d["weighted avg"]["support"]
                report_dict_dl_std["weighted avg"]["support"].append(d["weighted avg"]["support"])
                report_dict_dl["samples avg"]["precision"] += d["samples avg"]["precision"]
                report_dict_dl_std["samples avg"]["precision"].append(d["samples avg"]["precision"])
                report_dict_dl["samples avg"]["recall"] += d["samples avg"]["recall"]
                report_dict_dl_std["samples avg"]["recall"].append(d["samples avg"]["recall"])
                report_dict_dl["samples avg"]["f1-score"] += d["samples avg"]["f1-score"]
                report_dict_dl_std["samples avg"]["f1-score"].append(d["samples avg"]["f1-score"])
                report_dict_dl["samples avg"]["support"] += d["samples avg"]["support"]
                report_dict_dl_std["samples avg"]["support"].append(d["samples avg"]["support"])
                report_dict_dl["accuracy_score"] += d["accuracy_score"]
                report_dict_dl_std["accuracy_score"].append(d["accuracy_score"])
                count_r_dl += 1
        elif "report_ximl" in json_file:
            with open("./runs/" + json_file) as json_data:
                d = json.load(json_data)
                for i in range(11):
                    report_dict_ximl[str(i)]["precision"] += d[str(i)]["precision"]
                    report_dict_ximl_std[str(i)]["precision"].append(d[str(i)]["precision"])
                    report_dict_ximl[str(i)]["recall"] += d[str(i)]["recall"]
                    report_dict_ximl_std[str(i)]["recall"].append(d[str(i)]["recall"])
                    report_dict_ximl[str(i)]["f1-score"] += d[str(i)]["f1-score"]
                    report_dict_ximl_std[str(i)]["f1-score"].append(d[str(i)]["f1-score"])
                    report_dict_ximl[str(i)]["support"] += d[str(i)]["support"]
                    report_dict_ximl_std[str(i)]["support"].append(d[str(i)]["support"])
                report_dict_ximl["macro avg"]["precision"] += d["macro avg"]["precision"]
                report_dict_ximl_std["macro avg"]["precision"].append(d["macro avg"]["precision"])
                report_dict_ximl["macro avg"]["recall"] += d["macro avg"]["recall"]
                report_dict_ximl_std["macro avg"]["recall"].append(d["macro avg"]["recall"])
                report_dict_ximl["macro avg"]["f1-score"] += d["macro avg"]["f1-score"]
                report_dict_ximl_std["macro avg"]["f1-score"].append(d["macro avg"]["f1-score"])
                report_dict_ximl["macro avg"]["support"] += d["macro avg"]["support"]
                report_dict_ximl_std["macro avg"]["support"].append(d["macro avg"]["support"])
                report_dict_ximl["weighted avg"]["precision"] += d["weighted avg"]["precision"]
                report_dict_ximl_std["weighted avg"]["precision"].append(d["weighted avg"]["precision"])
                report_dict_ximl["weighted avg"]["recall"] += d["weighted avg"]["recall"]
                report_dict_ximl_std["weighted avg"]["recall"].append(d["weighted avg"]["recall"])
                report_dict_ximl["weighted avg"]["f1-score"] += d["weighted avg"]["f1-score"]
                report_dict_ximl_std["weighted avg"]["f1-score"].append(d["weighted avg"]["f1-score"])
                report_dict_ximl["weighted avg"]["support"] += d["weighted avg"]["support"]
                report_dict_ximl_std["weighted avg"]["support"].append(d["weighted avg"]["support"])
                report_dict_ximl["samples avg"]["precision"] += d["samples avg"]["precision"]
                report_dict_ximl_std["samples avg"]["precision"].append(d["samples avg"]["precision"])
                report_dict_ximl["samples avg"]["recall"] += d["samples avg"]["recall"]
                report_dict_ximl_std["samples avg"]["recall"].append(d["samples avg"]["recall"])
                report_dict_ximl["samples avg"]["f1-score"] += d["samples avg"]["f1-score"]
                report_dict_ximl_std["samples avg"]["f1-score"].append(d["samples avg"]["f1-score"])
                report_dict_ximl["samples avg"]["support"] += d["samples avg"]["support"]
                report_dict_ximl_std["samples avg"]["support"].append(d["samples avg"]["support"])
                report_dict_ximl["accuracy_score"] += d["accuracy_score"]
                report_dict_ximl_std["accuracy_score"].append(d["accuracy_score"])
                count_r_ximl += 1
        else:
            pass
    explanation_score_dict_dl_mean = {
        k: v / count_dl for k, v in explanation_score_dict_dl_mean.items()
    }
    explanation_score_dict_ximl_mean = {
        k: v / count_ximl for k, v in explanation_score_dict_ximl_mean.items()
    }
    with open("./runs/explanation_score_dl_mean_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(explanation_score_dict_dl_mean, f, indent=2)
    with open("./runs/explanation_score_ximl_mean_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(explanation_score_dict_ximl_mean, f, indent=2)

    explanation_score_dict_dl_std[
        "high_activated_right_pixels_std"
    ] = np.std(np.array(score_dl_std_list[0]))
    explanation_score_dict_dl_std[
        "high_activated_wrong_pixels_std"
    ] = np.std(np.array(score_dl_std_list[1]))
    explanation_score_dict_dl_std[
        "explanation_score_std"
    ] = np.std(np.array(score_dl_std_list[2]))

    explanation_score_dict_ximl_std[
        "high_activated_right_pixels_std"
    ] = np.std(np.array(score_ximl_std_list[0]))
    explanation_score_dict_ximl_std[
        "high_activated_wrong_pixels_std"
    ] = np.std(np.array(score_ximl_std_list[1]))
    explanation_score_dict_ximl_std[
        "explanation_score_std"
    ] = np.std(np.array(score_ximl_std_list[2]))
    with open("./runs/explanation_score_dl_std_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(explanation_score_dict_dl_std, f, indent=2)
    with open("./runs/explanation_score_ximl_std_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(explanation_score_dict_ximl_std, f, indent=2)

    for k, v in report_dict_dl.items():
        if k != "accuracy_score":
            for ki, vi in v.items():
                report_dict_dl[k][ki] /= count_r_dl
        else:
            report_dict_dl[k] /= count_r_dl
    with open("./runs/report_dl_mean_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(report_dict_dl, f, indent=2)

    for k, v in report_dict_ximl.items():
        if k != "accuracy_score":
            for ki, vi in v.items():
                report_dict_ximl[k][ki] /= count_r_ximl
        else:
            report_dict_ximl[k] /= count_r_ximl
    with open("./runs/report_ximl_mean_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(report_dict_ximl, f, indent=2)

    for k, v in report_dict_dl_std.items():
        if k != "accuracy_score":
            for ki, vi in v.items():
                report_dict_dl_std[k][ki] = np.std(np.array(report_dict_dl_std[k][ki]))
        else:
            report_dict_dl_std[k] = np.std(np.array(report_dict_dl_std[k]))
    with open("./runs/report_dl_std_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(report_dict_dl_std, f, indent=2)

    for k, v in report_dict_ximl_std.items():
        if k != "accuracy_score":
            for ki, vi in v.items():
                report_dict_ximl_std[k][ki] = np.std(np.array(report_dict_ximl_std[k][ki]))
        else:
            report_dict_ximl_std[k] = np.std(np.array(report_dict_ximl_std[k]))
    with open("./runs/report_ximl_std_" +
              dt.strftime(dt.now(), "%Y%m%d%H%M%S") + ".json", "w") as f:
        json.dump(report_dict_ximl_std, f, indent=2)
