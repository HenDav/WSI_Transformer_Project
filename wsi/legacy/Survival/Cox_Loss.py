import torch
import numpy as np
import inspect
import pandas as pd

def Cox_loss(risk_function_results: torch.Tensor, targets: torch.Tensor, censored: torch.Tensor) -> torch.float32:
    """
    This function implements the negative log partial likelihood used a loss.
    :param risk_function_results: risk feature results which is the result of beta_T * x, where x is the tile feature
    vector and beta is the coefficient vector.
    :param targets: targets representing the continous survival time
    :return:
    """
    num_tiles = targets.shape[0]
    # outer loop running over all live patients in the minibatch:
    # Actually we're going over all patients in the outer loop but in the inner loop we'll run over all patients still living
    # at the time the patient in the outer loop lives)
    '''
    loss, counter = 0, 0
    for i in range(num_tiles):
        if censored[i]:
            continue
        inner_loop_sum = 0
        for j in range(num_tiles):
            #  I'll assume that i in included in the inner summation
            # skipping patients j that died before patient i:
            if targets[j] < targets[i]:
                continue

            inner_loop_sum += torch.exp(risk_function_results[j])

        loss += risk_function_results[i] - torch.log(inner_loop_sum)
        counter += 1

    loss /= counter
    '''

    # Other calculation:
    order = reversed(np.argsort(targets.cpu()))  # indices to sort targets from largest to smallest (largest risk = survive less)

    risk_scores_s = risk_function_results[order]  # order the scores from smallest survival time to largest survival time
    censored_s = censored[order]

    cumsum_vec = torch.cumsum(torch.exp(risk_scores_s), dim=0)
    cumsum_vec_0 = cumsum_vec[censored_s == 0]
    risk_0 = risk_scores_s[censored_s == 0]
    likelihood_vec = risk_0 - torch.log(cumsum_vec_0)
    loss = torch.mean(likelihood_vec)


    if inspect.stack()[1][3] == 'train' and torch.isnan(loss):
        data_dict = {'Targets': list(targets.detach().cpu().numpy()),
                     'scores(risk_scores_s)': list(torch.reshape(risk_scores_s, (18,)).detach().cpu().numpy()),
                     'torch.exp(risk_scores_s)': list(torch.reshape(torch.exp(risk_scores_s), (18,)).detach().cpu().numpy()),
                     'Censored': list(censored_s.detach().cpu().numpy()),
                     'cumsum_vec_0': list(torch.reshape(cumsum_vec_0, (9,)).detach().cpu().numpy()) + [-1] * 9,
                     'torch.log(cumsum_vec_0)': list(torch.reshape(torch.log(cumsum_vec_0), (9,)).detach().cpu().numpy()) + [-1] * 9,
                     'risk_0': list(torch.reshape(risk_0, (9,)).detach().cpu().numpy()) + [-1] * 9
                     }

        DF = pd.DataFrame(data_dict)
        DF.to_excel('debug_data.xlsx')

        print('Loss is NaN. Here\'s some data:')
        print('targets: {}'.format(targets))
        print('scores(risk_scores_s): {}'.format(risk_scores_s))
        print('torch.exp(risk_scores_s): {}'.format(torch.exp(risk_scores_s)))
        print('Censored: {}'.format(censored_s))
        print('cumsum_vec_0: {}'.format(cumsum_vec_0))
        print('torch.log(cumsum_vec_0): {}'.format(torch.log(cumsum_vec_0)))
        print('risk_0: {}'.format(risk_0))
        raise Exception('found NANs')

    #print('Cox -> num of samples in use {}'.format(sum(censored_s == 0).item()))
    return -loss


def L2_Loss(model_outputs: torch.Tensor, targets: torch.Tensor, censored: torch.Tensor) -> torch.float32:
    # in case the model outputs has 2 channels it means that it came from a binary model. We need to turn it into 1
    # channel by softmaxing and taking the second channel
    if model_outputs.size(1) == 2:
        new_model_outputs = torch.nn.functional.softmax(model_outputs, dim=1)[:, 1]
    else:
        new_model_outputs = model_outputs

    '''for i in range(num_samples):
        if not censored[i] or (censored[i] and new_model_outputs[i] < targets[i]):
            loss_1 += (new_model_outputs[i] - targets[i]) ** 2'''

    valid_indices_1 = np.where(censored == False)[0]
    valid_indices_2 = np.where(new_model_outputs[:, 0].cpu() < targets.cpu())[0]
    valid_indices = list(set(np.concatenate((valid_indices_1, valid_indices_2))))
    #loss = torch.sum(torch.sqrt((new_model_outputs[valid_indices][:, 0] - targets[valid_indices]) ** 2))
    loss = torch.sum((new_model_outputs[valid_indices][:, 0] - targets[valid_indices]) ** 2)

    #print('L2 -> num of samples in use {}'.format(len(valid_indices)))
    return loss


def Combined_loss(model_outputs: torch.Tensor, targets_time: torch.Tensor, targets_binary: torch.Tensor, censored: torch.Tensor, weights: list = [1, 1, 1]):

    '''
    # Check that the model outputs and the targets are on the same device:
    is_cuda_model_outputs = model_outputs.is_cuda
    is_cuda_targets_time = targets_time.is_cuda
    is_cuda_targets_binary = targets_binary.is_cuda

    if is_cuda_model_outputs != is_cuda_targets_time:
        targets_time = targets_time.to('cuda')
    if is_cuda_model_outputs != is_cuda_targets_binary:
        targets_binary = targets_binary.to('cuda')
    '''

    # Compute Cox loss:
    loss_cox = Cox_loss(risk_function_results=model_outputs[:, 0], targets=targets_time, censored=censored)

    # Compute L2 loss
    loss_L2 = L2_Loss(model_outputs=torch.reshape(model_outputs[:, 1], (model_outputs[:, 1].size(0), 1)), targets=targets_time, censored=censored)
    # Compute Cross Entropy loss:
    valid_indices = targets_binary != -1
    if len(valid_indices.size()) == 2:
        valid_indices = torch.reshape(valid_indices, (valid_indices.size(0),))

    outputs_for_binary = model_outputs[valid_indices][:, 2:]

    targets_for_cross_entropy_calculation = targets_binary[valid_indices]
    if len(targets_for_cross_entropy_calculation.size()) == 2:
        targets_for_cross_entropy_calculation = torch.reshape(targets_for_cross_entropy_calculation, (targets_for_cross_entropy_calculation.size(0),))

    loss_cross_entropy = torch.nn.functional.cross_entropy(outputs_for_binary, targets_for_cross_entropy_calculation)
    # Combine together:
    total_loss = weights[0] * loss_cox\
                 + weights[1] * loss_L2\
                 + weights[2] * loss_cross_entropy

    return total_loss, loss_cox,  loss_L2, loss_cross_entropy


