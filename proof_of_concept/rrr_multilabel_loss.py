import copy
import torch
import torch.nn as nn
from resnet18 import ResNet
from model import MultiLabelClassifier
from captum.attr import IntegratedGradients

prediction_loss_function = nn.BCEWithLogitsLoss()


def _rrr_multilabel_loss_v1(model, sample,
                            prediction, target_vector,
                            target_1, target_2,
                            mask_1, mask_2):
    _model_1 = MultiLabelClassifier()
    _model_2 = MultiLabelClassifier()
    _model_1.load_state_dict(model.state_dict())
    _model_2.load_state_dict(model.state_dict())
    _optimizer_1 = torch.optim.Adam(_model_1.parameters(), lr=1e-3)
    _optimizer_1.zero_grad()
    _optimizer_2 = torch.optim.Adam(_model_2.parameters(), lr=1e-3)
    _optimizer_2.zero_grad()
    full_mask = torch.ones_like(sample)
    prediction_loss = prediction_loss_function(prediction, target_vector)
    integrated_gradient_1 = IntegratedGradients(_model_1)
    gradient_first_digit = integrated_gradient_1.attribute(sample, target=target_1)
    gradient_first_digit[gradient_first_digit < 0] = 0
    gradient_counterfactual_1 = integrated_gradient_1.attribute(sample, target=target_2)
    gradient_counterfactual_1[gradient_counterfactual_1 < 0] = 0
    integrated_gradient_2 = IntegratedGradients(_model_2)
    gradient_second_digit = integrated_gradient_2.attribute(sample, target=target_2 + 10)
    gradient_second_digit[gradient_second_digit < 0] = 0
    gradient_counterfactual_2 = integrated_gradient_1.attribute(sample, target=10 + target_1)
    gradient_counterfactual_2[gradient_counterfactual_2 < 0] = 0
    explanation_loss = torch.sum(gradient_first_digit * mask_1 +
                                 gradient_second_digit * mask_2 +
                                 gradient_counterfactual_1 * full_mask +
                                 gradient_counterfactual_2 * full_mask) / len(gradient_first_digit)
    loss = prediction_loss + explanation_loss

    return loss, prediction_loss


def rrr_multilabel_loss_v2(model, sample,
                           prediction, target_vector,
                           target_1, target_2,
                           mask_1, mask_2,
                           device):
    _model = MultiLabelClassifier().to(device)
    _model.load_state_dict(model.state_dict())
    _optimizer = torch.optim.Adam(_model.parameters(), lr=1e-3)
    _optimizer.zero_grad()
    full_mask = torch.ones_like(sample).to(device)
    prediction_loss = prediction_loss_function(prediction, target_vector)
    integrated_gradient = IntegratedGradients(_model)
    explanation_loss = torch.zeros_like(sample).to(device)
    for i in range(len(prediction[-1])):
        gradient_tensor = integrated_gradient.attribute(sample, target=i).to(device)
        gradient_tensor[gradient_tensor < 0] = 0
        if i < 10:
            explanation_loss[target_1 == i] += \
                gradient_tensor[target_1 == i] * mask_1[target_1 == i]
            explanation_loss[target_1 != i] += \
                gradient_tensor[target_1 != i] * full_mask[target_1 != i]
        else:
            explanation_loss[target_2 == i - 10] += \
                gradient_tensor[target_2 == i - 10] * mask_2[target_2 == i - 10]
            explanation_loss[target_2 != i - 10] += \
                gradient_tensor[target_2 != i - 10] * full_mask[target_2 != i - 10]

    loss = prediction_loss + torch.sum(explanation_loss) / len(sample)

    return loss, prediction_loss


def rrr_multilabel_loss(model, sample,
                        prediction, target_vector,
                        target_1, target_2,
                        mask_1, mask_2,
                        device):
    _model = MultiLabelClassifier().to(device)
    _model.load_state_dict(model.state_dict())
    prediction_loss = prediction_loss_function(prediction, target_vector)
    integrated_gradient = IntegratedGradients(_model)

    gradient_tensor_1 = integrated_gradient.attribute(sample, target=target_1).to(device)
    gradient_tensor_1[gradient_tensor_1 < 0] = 0
    gradient_tensor_2 = integrated_gradient.attribute(sample, target=target_2).to(device)
    gradient_tensor_2[gradient_tensor_2 < 0] = 0
    sigmoid = nn.Sigmoid()
    log_prob = (torch.log(sigmoid(prediction)) * target_vector).sum(-1)
    log_prob_matrix = torch.zeros_like(gradient_tensor_1)
    for i in range(len(gradient_tensor_1)):
        log_prob_matrix[i] = torch.ones_like(gradient_tensor_1)[i] * log_prob[i]

    explanation_loss = torch.sum((gradient_tensor_1 * mask_1 * log_prob_matrix)**2 +
                                 (gradient_tensor_2 * mask_2 * log_prob_matrix)**2)

    loss = prediction_loss + (explanation_loss / len(sample))

    return loss, prediction_loss, (explanation_loss / len(sample))


def rrr_loss(model, sample,
             prediction, target_vector,
             target_1, target_2,
             mask_1, mask_2,
             device):
    _model = ResNet().to(device)
    _model.load_state_dict(model.state_dict())
    prediction_loss = prediction_loss_function(prediction, target_vector)
    integrated_gradient = IntegratedGradients(_model)

    gradient_tensor_1 = integrated_gradient.attribute(sample, target=target_1).to(device)
    gradient_tensor_1_positive = copy.deepcopy(gradient_tensor_1)
    gradient_tensor_1_positive[gradient_tensor_1_positive < 0] = 0
    gradient_tensor_1_negative = copy.deepcopy(gradient_tensor_1)
    gradient_tensor_1_negative[gradient_tensor_1_negative > 0] = 0
    gradient_tensor_2 = integrated_gradient.attribute(sample, target=target_2).to(device)
    gradient_tensor_2_positive = copy.deepcopy(gradient_tensor_2)
    gradient_tensor_2_positive[gradient_tensor_2_positive < 0] = 0
    gradient_tensor_2_negative = copy.deepcopy(gradient_tensor_2)
    gradient_tensor_2_negative[gradient_tensor_2_negative > 0] = 0
    sigmoid = nn.Sigmoid()
    log_prob = (torch.log(sigmoid(prediction)) * target_vector).sum(-1)
    log_prob_matrix = torch.zeros_like(gradient_tensor_1)
    for i in range(len(gradient_tensor_1)):
        log_prob_matrix[i] = torch.ones_like(gradient_tensor_1)[i] * log_prob[i]

    explanation_loss = torch.sum((gradient_tensor_1_positive * mask_1 * log_prob_matrix)**2 +
                                 (gradient_tensor_1_negative * mask_2 * log_prob_matrix)**2 +
                                 (gradient_tensor_2_positive * mask_2 * log_prob_matrix)**2 +
                                 (gradient_tensor_2_negative * mask_1 * log_prob_matrix)**2)

    loss = prediction_loss + 1e-1 * explanation_loss

    return loss, prediction_loss, 1e-1 * explanation_loss


def rrr(model, sample,
        prediction, target_vector,
        target_1, target_2,
        mask_1, mask_2,
        device, model_type):
    if model_type == 'MLC':
        _model = MultiLabelClassifier().to(device)
    else:
        _model = ResNet().to(device)
    _model.load_state_dict(model.state_dict())
    prediction_loss = prediction_loss_function(prediction, target_vector)
    integrated_gradient = IntegratedGradients(_model)

    gradient_tensor_1 = integrated_gradient.attribute(sample, target=target_1).to(device)
    gradient_tensor_1_positive = copy.deepcopy(gradient_tensor_1)
    gradient_tensor_1_positive[gradient_tensor_1_positive < 0] = 0
    gradient_tensor_1_negative = copy.deepcopy(gradient_tensor_1)
    gradient_tensor_1_negative[gradient_tensor_1_negative > 0] = 0
    gradient_tensor_2 = integrated_gradient.attribute(sample, target=target_2).to(device)
    gradient_tensor_2_positive = copy.deepcopy(gradient_tensor_2)
    gradient_tensor_2_positive[gradient_tensor_2_positive < 0] = 0
    gradient_tensor_2_negative = copy.deepcopy(gradient_tensor_2)
    gradient_tensor_2_negative[gradient_tensor_2_negative > 0] = 0
    sigmoid = nn.Sigmoid()
    target_1_vector = torch.nn.functional.one_hot(target_1, num_classes=11)
    log_prob_1 = (torch.log(sigmoid(prediction)) * target_1_vector).sum(-1)
    log_prob_matrix_1 = torch.zeros_like(gradient_tensor_1)
    for i in range(len(gradient_tensor_1)):
        log_prob_matrix_1[i] = torch.ones_like(gradient_tensor_1)[i] * log_prob_1[i]
    target_2_vector = torch.nn.functional.one_hot(target_2, num_classes=11)
    log_prob_2 = (torch.log(sigmoid(prediction)) * target_2_vector).sum(-1)
    log_prob_matrix_2 = torch.zeros_like(gradient_tensor_1)
    for i in range(len(gradient_tensor_1)):
        log_prob_matrix_2[i] = torch.ones_like(gradient_tensor_1)[i] * log_prob_2[i]

    explanation_loss = torch.sum((gradient_tensor_1_positive * mask_1 * log_prob_matrix_1)**2 +
                                 (gradient_tensor_1_negative * mask_2 * log_prob_matrix_1)**2 +
                                 (gradient_tensor_2_positive * mask_2 * log_prob_matrix_2)**2 +
                                 (gradient_tensor_2_negative * mask_1 * log_prob_matrix_2)**2)

    loss = prediction_loss + 1e-1 * (explanation_loss / len(sample))

    return loss, prediction_loss, 1e-1 * (explanation_loss / len(sample))
