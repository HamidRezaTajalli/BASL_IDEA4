from pathlib import Path
import torch
import models
from dataset_handler import emnist, fmnist, mnist, cifar10
from train_and_validation import sl_simple

torch.manual_seed(47)
import numpy as np
np.random.seed(47)
def nc_test(tp_name, dataset, arch_name, cut_layer, base_path, exp_num, batch_size, alpha_fixed,
            num_clients, bd_label, tb_inj, initial_alpha):
    experiment_name = f"{tp_name}_exp{exp_num}_{dataset}_{arch_name}_{num_clients}_{cut_layer}_{tb_inj}_{alpha_fixed}_{initial_alpha}"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if arch_name.lower() == 'stripnet':
        if cut_layer == 3:
            cut_layer = 4

    # creating path object for saved client and server model

    client_path = base_path.joinpath(f'client_{experiment_name}.pt')
    server_path = base_path.joinpath(f'server_{experiment_name}.pt')
    whole_path = base_path.joinpath(f'wholemodel_{experiment_name}.pt')

    if not client_path.exists():
        raise Exception(f"Client path {client_path} does not exist")
    if not server_path.exists():
        raise Exception(f"Server path {server_path} does not exist")

    # creating the model objects and then loading the state dicts from the path address

    # client_model = models.get_model(arch_name=arch_name, dataset=dataset, model_type='client',
    #                                 cut_layer=cut_layer).to(device)
    # server_model = models.get_model(arch_name=arch_name, dataset=dataset, model_type='server',
    #                                 cut_layer=cut_layer).to(device)
    whole_model = models.get_model(arch_name=arch_name, dataset=dataset, model_type='whole',
                                   cut_layer=cut_layer).to(device)

    client_state_dict = torch.load(client_path, map_location=device)
    server_state_dict = torch.load(server_path, map_location=device)

    # client_model.load_state_dict(client_state_dict)
    # server_model.load_state_dict(server_state_dict)
    whole_model.load_state_dict(server_state_dict)


    # loading the lower half of the whole model with client parameters

    for name, param in client_state_dict.items():
        if name.split('.')[0].split('_')[-1].isdigit():
            layer_num = int(name.split('.')[0].split('_')[-1])
        elif name.split('.')[1].isdigit():
            layer_num = int(name.split('.')[1].split('_')[-1])
        else:
            continue
        if layer_num <= cut_layer and name in whole_model.state_dict():
            whole_model.state_dict()[name].copy_(param)

    # for name, param in client_model.state_dict().items():
    #     print(name)
    # print('-----------------' * 5)
    # for name, param in client_model.named_parameters():
    #     print(name)
    # print('-----------------' * 5)
    # for name, param in client_state_dict.items():
    #     print(name)

    # saving the whole model
    torch.save(whole_model.state_dict(), whole_path)


    # checking to see if the constructed model works fine with dataset

    ds_load_dict = {'cifar10': cifar10, 'fmnist': fmnist, 'mnist': mnist, 'emnist': emnist}
    dataloaders, classes_names = ds_load_dict[dataset].get_dataloaders_simple(
        batch_size=batch_size, train_ds_num=1 , drop_last=False, is_shuffle=True)
    phase = 'test'
    whole_model.eval()
    epoch_loss = 0.0
    epoch_corrects = 0.0
    with torch.no_grad():
        for batch_num, data in enumerate(dataloaders[phase]):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = whole_model(inputs)
            output_preds = torch.max(outputs, dim=1)
            corrects = torch.sum(output_preds[1] == labels).item()
            epoch_corrects += corrects

    epoch_corrects = (epoch_corrects / len(dataloaders[phase].dataset)) * 100
    print_string = f"[{phase} accuracies: {epoch_corrects:>6}]"
    print(print_string)















datasets = ['mnist']
models_list = ['stripnet']
num_of_exp = 1
cut_layers = [1]
num_clients_list = [3]
tb_inj = False
alpha_list = [0.04]
save_path = '.'
tp_name = 'BASL_IDEA4'
batch_size = 128
alpha_fixed = True
bd_label = 0
base_path = Path()

for dataset in datasets:
    for arch_name in models_list:
        for num_clients in num_clients_list:
            for cut_layer in cut_layers:
                for exp_num in range(num_of_exp):
                    for alpha in alpha_list:
                        nc_test(tp_name=tp_name, dataset=dataset, arch_name=arch_name,
                                cut_layer=cut_layer,
                                base_path=base_path, exp_num=exp_num, batch_size=batch_size,
                                alpha_fixed=alpha_fixed,
                                num_clients=num_clients, bd_label=bd_label, tb_inj=tb_inj,
                                initial_alpha=alpha)
