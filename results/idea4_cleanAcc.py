import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib as mpl
from heatmap_helper import heatmap, annotate_heatmap

parser = argparse.ArgumentParser()
parser.add_argument("--loadpath", type=str, default='.')
# parser.add_argument("--dataset", type=str, default='cifar10', help='mnist, fmnist, cifar10',
#                     choices=['mnist', 'fmnist', 'cifar10'], required=True, )
parser.add_argument("--savepath", type=str, default='.')
args = parser.parse_args()


def main():
    plt.rcParams["font.family"] = "Times New Roman"

    # Read the reuslts from the csv file
    loadpath = Path(args.loadpath)
    path_save = Path(args.savepath)

    models = ['resnet9', 'lenet', 'stripnet']
    model_labels = ['ResNet', 'LeNet', 'StripNet']
    legend_style_labels = ['CDA', 'ASR']
    datasets = ['mnist', 'fmnist', 'cifar10']
    cut_layers = [1, 2, 3]
    num_clients_list = [1, 3, 5, 7]
    colors = ['blue', 'red', 'green']
    cda_markers = ['o', 'D', 'x']
    asr_markers = ['*', '', 'X']
    markers = ['x', 'X']
    linestyles = ['dotted', 'solid']
    n_experiments = 1

    fig, axs = plt.subplots(nrows=len(datasets), ncols=len(
        models), figsize=(5, 6), sharex=True, sharey=True, constrained_layout=True)

    col_leg_list = []

    df = pd.read_csv(loadpath)
    vmin, vmax = df['TEST_ACCURACY'].min(), df['TEST_ACCURACY'].max()



    column = 0
    row = 0
    for idx, ax in enumerate(axs.flat):
        if column >= len(cut_layers):
            column = 0
            row += 1

        model = models[column]
        dataset = datasets[row]

        np_array = np.empty(shape=(len(num_clients_list), len(cut_layers)), dtype=np.float64)
        for dim1, num_clients in enumerate(num_clients_list):
            for dim2, cut_layer in enumerate(cut_layers):
                slctd_df = df[(df['NETWORK_ARCH'] == model) &
                              (df['DATASET'] == dataset) &
                              (df['NUMBER_OF_CLIENTS'] == num_clients) &
                              (df['CUT_LAYER'] == cut_layer)]


                if not len(slctd_df['TEST_ACCURACY'].values) > 0:
                    print(num_clients, dataset, model, cut_layer)
                    raise Exception('above row not found in pandas datafram.')
                else:
                    np_array[dim1, dim2] = slctd_df['TEST_ACCURACY'].values[0]


            # clnums_CDA_list = [item * 100 for item in clnums_CDA_list]
            # clnums_ASR_list = [item * 100 for item in clnums_ASR_list]

            # mean = np.mean(list_frzlyr_asr, axis=1) * 100
            # min = np.min(list_frzlyr_asr, axis=1) * 100
            # max = np.max(list_frzlyr_asr, axis=1) * 100

            # err = np.array([mean - min, max - mean])
        np_array = np.around(np_array, decimals=2)
        im = heatmap(data=np_array, row_labels=num_clients_list, col_labels=cut_layers, ax=ax, vmin=vmin, vmax=vmax)
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
        # ax.imshow(np_array)
        # for i in range(len(num_clients_list)):
        #     for j in range(len(cut_layers)):
        #         text = ax.text(j, i, np_array[i, j],
        #                        ha="center", va="center", color="w")
        # ax.errorbar(x=num_clients_list, y=clnums_CDA_list, marker=markers[0], alpha=0.8, markersize=4,
        #             linestyle=linestyles[0], color=colors[models.index(model)])
        # line_handle = ax.errorbar(x=num_clients_list, y=clnums_ASR_list, marker=markers[1], alpha=0.8, markersize=4,
        #                           label=legend_color_labels[models.index(model)], linestyle=linestyles[1],
        #                           color=colors[models.index(model)])
        # col_leg_list.append(line_handle)

        column += 1
    cax, kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    fig.colorbar(im, cax=cax, **kw)

    # Set the labels
    for ax, col in zip(axs[0], model_labels):
        ax.set_title(f'Model = {col}')

    for ax, row in zip(axs[:, 0], datasets):
        ax.set_ylabel('Dataset' + f' = {row}', rotation=90, size='large')

    # styl_leg_list = [mlines.Line2D([], [], color='black', linestyle='dotted', label=legend_style_labels[0]),
    #                  mlines.Line2D([], [], color='black', linestyle='solid', label=legend_style_labels[1])]

    # Set the legend showing the models with the corresponding marker
    # handles, labels = axs[0, 0].get_legend_handles_labels()
    # handles = col_leg_list + styl_leg_list
    # fig.legend(handles=handles, bbox_to_anchor=(0.75, 0.078), fancybox=False, shadow=False, ncol=len(handles))
    # fig.legend(handles,
    #            legend_labels,
    #            bbox_to_anchor=(0.65, 0.095), fancybox=False, shadow=False, ncol=len(colors))

    # Set the x and y labels
    fig.supxlabel('CutLayer')
    fig.supylabel('Client Numbers')

    # Set the ticks
    for ax in axs.flat:
        ax.set_xticks(np.arange(len(cut_layers)), labels=cut_layers)
        ax.set_yticks(np.arange(len(num_clients_list)), labels=num_clients_list)
        # ax.set_yticks(num_clients_list)

        # for label in ax.get_xticklabels()[::2]:
        #     label.set_visible(False)

    # Set the grid
    sns.despine(left=True)
    # plt.tight_layout()
    # path_save = os.path.join(
    #     path_save, f'rate_vs_size_{args.trigger_color}_{args.pos}.pdf')
    # plt.savefig(path_save)
    plt.savefig(path_save.joinpath(f'idea4_cleanAcc.pdf'))
    # plt.show()


if __name__ == '__main__':
    main()
