from pathlib import Path
import torch
import gc
import csv
import pickle

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchsummary import summary

from helper import EarlyStopping
import models
from dataset_handler import emnist, fmnist, mnist, cifar10
torch.manual_seed(47)
import numpy as np
np.random.seed(47)

def tr_lr_test(tp_name, dataset, arch_name, cut_layer, base_path, exp_num, batch_size, alpha_fixed,
            num_clients, bd_label, tb_inj, initial_alpha):
    experiment_name = f"{tp_name}_exp{exp_num}_{dataset}_{arch_name}_{num_clients}_{cut_layer}_{tb_inj}_{alpha_fixed}_{initial_alpha}"
    whole_model_address = f"wholemodel_{tp_name}_exp{exp_num}_mnist_{arch_name}_{num_clients}_{cut_layer}_{tb_inj}_{alpha_fixed}_{initial_alpha}.pt"
    trlrnd_address = f"trlrnd_{tp_name}_exp{exp_num}_mnist_{arch_name}_{num_clients}_{cut_layer}_{tb_inj}_{alpha_fixed}_{initial_alpha}.pt"
    trigger_address = f"trigger_obj.pkl"
    trigger_path = base_path.joinpath(trigger_address)

    with open(file=trigger_path, mode='rb') as file:
        trigger_obj = pickle.load(file)



    csv_path = base_path.joinpath('transfer_results.csv')
    if not csv_path.exists():
        csv_path.touch()
        with open(file=csv_path, mode='w') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['EXPERIMENT_NUMBER', 'NETWORK_ARCH',
                                 'DATASET', 'NUMBER_OF_CLIENTS', 'CUT_LAYER', 'TB_INJECT', 'FIXED_ALPHA', 'ALPHA',
                                 'TRAIN_ACCURACY', 'TEST_ACCURACY', 'BD_TEST_ACCURACY'])


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    wholemodel_path = base_path.joinpath(whole_model_address)
    trlrnd_path = base_path.joinpath(trlrnd_address)
    if not wholemodel_path.exists():
        raise Exception(f"model path {wholemodel_path} does not exist")

    whole_model_state_dict = torch.load(wholemodel_path, map_location=device)
    whole_model = models.get_model(arch_name=arch_name, dataset=dataset, model_type='whole',
                                   cut_layer=cut_layer).to(device)

    whole_model.load_state_dict(whole_model_state_dict)

    # Freezing the model parameters exepct the last layer
    for layer in whole_model.layers[:-1]:
        for param in layer.parameters():
            param.requires_grad = False


    # Loading the dataset for training
    ds_load_dict = {'fmnist': fmnist, 'emnist': emnist, 'mnist': mnist}
    dataloaders, classes_names = ds_load_dict[dataset].get_dataloaders_normal(batch_size=batch_size, drop_last=False,
                                                                              is_shuffle=True, trigger_obj=trigger_obj,
                                                                              target_label=bd_label)

    # Training the last layer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=whole_model.parameters(), weight_decay=1e-4)

    # Metrics to store for training
    train_losses = []
    train_accuracies = []



    num_epochs = 60
    for epoch in range(num_epochs):
        whole_model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = whole_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(dataloaders['train']))
        train_accuracies.append(100 * correct / total)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%")



    test_accuracies = []
    test_losses = []

    # Clean Test loop
    whole_model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = whole_model(inputs)
            val_loss = criterion(outputs, labels)

            test_running_loss += val_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_losses.append(test_running_loss / len(dataloaders['test']))
    test_accuracies.append(100 * test_correct / test_total)

    print(f"Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}%")

    torch.save(whole_model.state_dict(), trlrnd_path)


    #Testing backdoor attacks

    bd_test_accuracies = []
    bd_test_losses = []

    # Backdoor Test loop
    whole_model.eval()
    bd_test_running_loss = 0.0
    bd_test_correct = 0
    bd_test_total = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['bd_test']:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = whole_model(inputs)
            val_loss = criterion(outputs, labels)

            bd_test_running_loss += val_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            bd_test_total += labels.size(0)
            bd_test_correct += (predicted == labels).sum().item()

    bd_test_losses.append(bd_test_running_loss / len(dataloaders['bd_test']))
    bd_test_accuracies.append(100 * bd_test_correct / bd_test_total)

    print(f"Backdoor Test Loss: {bd_test_losses[-1]:.4f}, Backdoor Test Accuracy: {bd_test_accuracies[-1]:.2f}%")

    with open(file=csv_path, mode='a') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [exp_num, arch_name, dataset, num_clients, cut_layer, tb_inj, alpha_fixed, initial_alpha,
             train_accuracies[-1],
             test_accuracies[-1],
             bd_test_accuracies[-1]])









