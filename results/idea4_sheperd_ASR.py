import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib as mpl


parser = argparse.ArgumentParser()
parser.add_argument("--loadpath", type=str, default='.')
# parser.add_argument("--dataset", type=str, default='cifar10', help='mnist, fmnist, cifar10',
#                     choices=['mnist', 'fmnist', 'cifar10'], required=True, )
parser.add_argument("--savepath", type=str, default='.')
args = parser.parse_args()


def main():
    sns.set()
    sns.set_theme()
    # sns.set_theme(rc={"figure.subplot.wspace": 0.002, "figure.subplot.hspace": 0.002})
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 22
    # sns.set_theme(style="whitegrid", font_scale=1.2)


    # Read the reuslts from the csv file
    loadpath = Path(args.loadpath)
    path_save = Path(args.savepath)

    # models = ['resnet9', 'lenet', 'stripnet']
    legend_color_labels = ['MNIST', 'FMNIST', 'EMNIST']
    legend_style_labels = ['CDA', 'ASR']
    datasets = ['mnist', 'fmnist', 'emnist']
    main_model = 'stripnet'
    cut_layers = [1]
    num_clients_list = [1, 3, 5, 7]
    alpha_list = [0.5, 0.09, 0.04]
    colors = ['blue', 'red', 'green']
    cda_markers = ['o', 'D', 'x']
    asr_markers = ['*', '', 'X']
    markers = ['x', 'X']
    linestyles = ['dotted', 'solid']
    n_experiments = 1

    fig, axs = plt.subplots(nrows=len(alpha_list), ncols=len(
        cut_layers), figsize=(6, 10), sharex=True, sharey=True)
    # fig.subplots_adjust(wspace=0.01, hspace=0.01)

    col_leg_list = []
    styl_leg_list = []

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

        for dataset in datasets:
            clnums_ASR_list = []
            clnums_CDA_list = []
            for num_clients in num_clients_list:
                slctd_df = df[(df['NETWORK_ARCH'] == main_model) &
                              (df['DATASET'] == dataset) &
                              (df['NUMBER_OF_CLIENTS'] == num_clients) &
                              (df['CUT_LAYER'] == cut_layer) &
                              (df['ALPHA'] == alpha)]

                if not len(slctd_df['TEST_ACCURACY'].values) > 0 or not len(slctd_df['BD_TEST_ACCURACY'].values) > 0:
                    print(num_clients, dataset, main_model, alpha, cut_layer)
                    raise Exception('above row not found in pandas datafram.')
                else:
                    clnums_CDA_list.append(slctd_df['TEST_ACCURACY'].values[0])
                    clnums_ASR_list.append(slctd_df['BD_TEST_ACCURACY'].values[0])
            # clnums_CDA_list = [item * 100 for item in clnums_CDA_list]
            # clnums_ASR_list = [item * 100 for item in clnums_ASR_list]

            # mean = np.mean(list_frzlyr_asr, axis=1) * 100
            # min = np.min(list_frzlyr_asr, axis=1) * 100
            # max = np.max(list_frzlyr_asr, axis=1) * 100

            # err = np.array([mean - min, max - mean])
            ax.errorbar(x=num_clients_list, y=clnums_CDA_list, marker=markers[0], alpha=0.8, markersize=4,
                        linestyle=linestyles[0], color=colors[datasets.index(dataset)])
            line_handle = ax.errorbar(x=num_clients_list, y=clnums_ASR_list, marker=markers[1], alpha=0.8, markersize=4,
                        label=legend_color_labels[datasets.index(dataset)], linestyle=linestyles[1], color=colors[datasets.index(dataset)])
            col_leg_list.append(line_handle)

        column += 1
    col_leg_list = col_leg_list[:len(datasets)]

    # Set the labels
    # for ax, col in zip(axs[0], cut_layers):
    #     ax.set_title(f'Cut Layer = {col}', fontsize=20)
    axs[0].set_title(f'Cut Layer = {cut_layers[0]}', fontsize=20)


    for ax, row in zip(axs[:], alpha_list):
        ax.set_ylabel(r'$\alpha$' + f' = {row}', rotation=90, size='large')

    styl_leg_list = [mlines.Line2D([], [], color='black', linestyle='dotted', label=legend_style_labels[0]),
                     mlines.Line2D([], [], color='black', linestyle='solid', label=legend_style_labels[1])]

    # Set the legend showing the models with the corresponding marker
    # handles, labels = axs[0, 0].get_legend_handles_labels()
    handles = col_leg_list + styl_leg_list
    fig.legend(handles=handles, bbox_to_anchor=(0.92, 0.091), fancybox=False, shadow=False, ncol=len(handles), prop={'size': 9})
    # fig.legend(handles,
    #            legend_labels,
    #            bbox_to_anchor=(0.65, 0.095), fancybox=False, shadow=False, ncol=len(colors))

    # Set the x and y labels
    fig.supxlabel('Number of Clients')
    fig.supylabel('ASR & CDA (%)')

    # Set the ticks
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xticks(num_clients_list)
        ax.set_yticks(np.arange(0, 120, 20))
        ax.set_ylim(-20, 120)
        ax.set_xlim(0, num_clients_list[-1] + 1)
        # for label in ax.get_xticklabels()[::2]:
        #     label.set_visible(False)
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)

    # Set the grid
    sns.despine(left=True)
    plt.tight_layout()
    # path_save = os.path.join(
    #     path_save, f'rate_vs_size_{args.trigger_color}_{args.pos}.pdf')
    # plt.savefig(path_save)
    plt.savefig(path_save.joinpath(f'idea4_sheperd_{main_model}.pdf'))
    # plt.show()


if __name__ == '__main__':
    main()
