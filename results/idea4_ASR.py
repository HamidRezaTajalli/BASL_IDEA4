import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--loadpath", type=str, default='.')
parser.add_argument("--savepath", type=str, default='.')
args = parser.parse_args()


def main():
    sns.set()
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # Read the reuslts from the csv file
    loadpath = Path(args.loadpath)
    path_save = Path(args.savepath)

    models = ['ResNet9', 'LeNet', 'StripNet']
    model = 'alexnet'
    colors = ['random', 'white', 'black']
    # TODO: change the legend labels to khat dar va bi khat
    legend_labels = ['ResNet9_ASR', 'LeNet_ASR', 'StripNet_ASR', 'ResNet9_CDA', 'LeNet_CDA', 'StripNet_CDA']
    datasets = ['mnist', 'fmnist', 'cifar10']
    dataset = 'cifar10'
    cut_layers = [1, 2, 3]
    num_clients_list = [1, 3, 5, 7]
    alpha_list = [0.5, 0.2, 0.09, 0.06, 0.04]
    markers = ['o', 's', 'D', 'X']
    linestyles = ['-', '--', '-.', ':']
    n_experiments = 1

    fig, axs = plt.subplots(nrows=len(alpha_list), ncols=len(
        cut_layers), figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)

    df = pd.read_csv(loadpath)
    # print(df)
    column = 0
    row = 0
    for idx, ax in enumerate(axs.flat):
        if column >= len(cut_layers):
            column = 0
            row += 1

        cut_layer = cut_layers[column]
        alpha = alpha_list[row]

        for model in models:
            clnums_ASR_list = []
            clnums_CDA_list = []
            for num_clients in num_clients_list:
                slctd_df = df[(df['freeze_layer_number'] == frzlyr) &
                              (df['dataname'] == dataset) &
                              (df['model'] == model) &
                              (df['epsilon'] == epsilon) &
                              (df['trigger_size'] == trigger_size) &
                              (df['trigger_color'] == color) &
                              (df['pos'] == trig_pos)]

                if not len(slctd_df['test_acc_backdoor'].values) > 0:
                    print(frzlyr, dataset, model, epsilon, trigger_size, color, trig_pos)
                    raise Exception('above row not found in pandas datafram.')
                else:
                    list_frzlyr_asr.append(slctd_df['test_acc_backdoor'].values[0])
            list_frzlyr_asr = [item * 100 for item in list_frzlyr_asr]

            # mean = np.mean(list_frzlyr_asr, axis=1) * 100
            # min = np.min(list_frzlyr_asr, axis=1) * 100
            # max = np.max(list_frzlyr_asr, axis=1) * 100

            # err = np.array([mean - min, max - mean])
            ax.errorbar(frz_lyrs, list_frzlyr_asr, marker=markers[colors.index(color)], alpha=0.8, markersize=4,
                        label=color, linestyle=linestyles[colors.index(color)])

        column += 1

    # Set the labels
    for ax, col in zip(axs[0], trigger_size_values):
        ax.set_title('Trigger size = {}'.format(col))

    for ax, row in zip(axs[:, 0], epsilon_values):
        ax.set_ylabel(r'$\epsilon$'+ f' = {row}', rotation=90, size='large')

    # Set the legend showing the models with the corresponding marker
    handles, labels = axs[0, 0].get_legend_handles_labels()

    fig.legend(handles,
               legend_labels,
               bbox_to_anchor=(0.65, 0.095), fancybox=False, shadow=False, ncol=len(colors))


    # Set the x and y labels
    fig.supxlabel('Freeze Layer (parameter#)')
    fig.supylabel('ASR (%)')

    # Set the ticks
    for ax in axs.flat:
        ax.set_xticks(frz_lyrs, labelsize=00.2)
        ax.set_yticks(np.arange(0, 110, 20))
        ax.set_ylim(0, 120)
        ax.set_xlim(max(frz_lyrs), min(frz_lyrs))
        for label in ax.get_xticklabels()[::2]:
            label.set_visible(False)

    # Set the grid
    sns.despine(left=True)
    plt.tight_layout()
    # path_save = os.path.join(
    #     path_save, f'rate_vs_size_{args.trigger_color}_{args.pos}.pdf')
    # plt.savefig(path_save)
    plt.savefig(path_save.joinpath(f'alexnet_{dataset}_{trig_pos}.pdf'))
    # plt.show()


if __name__ == '__main__':
    main()
