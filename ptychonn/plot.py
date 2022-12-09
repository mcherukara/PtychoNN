import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics: dict,
                 save_fname: str = None,
                 show_fig: bool = False):

    losses_arr = np.array(metrics['losses'])
    val_losses_arr = np.array(metrics['val_losses'])
    print("Shape of losses array is ", losses_arr.shape)
    fig, ax = plt.subplots(3, sharex=True, figsize=(15, 8))
    ax[0].plot(losses_arr[1:, 0], 'C3o-', label="Train")
    ax[0].plot(val_losses_arr[1:, 0], 'C0o-', label="Val")
    ax[0].set(ylabel='Loss')
    ax[0].set_yscale('log')
    ax[0].grid()
    ax[0].legend(loc='center right')
    ax[0].set_title('Total loss')

    ax[1].plot(losses_arr[1:, 1], 'C3o-', label="Train Amp loss")
    ax[1].plot(val_losses_arr[1:, 1], 'C0o-', label="Val Amp loss")
    ax[1].set(ylabel='Loss')
    ax[1].set_yscale('log')
    ax[1].grid()
    ax[1].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    ax[1].set_title('Phase loss')

    # ax[2].plot(losses_arr[1:, 2], 'C3o-', label="Train Ph loss")
    # ax[2].plot(val_losses_arr[1:, 2], 'C0o-', label="Val Ph loss")
    # ax[2].set(ylabel='Loss')
    # ax[2].grid()
    # ax[2].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    # ax[2].set_yscale('log')
    # ax[2].set_title('Mag los')

    plt.tight_layout()
    plt.xlabel("Epochs")

    if save_fname is not None:
        plt.savefig(save_fname)
    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_test_data(selected_diffs: np.ndarray,
                   selected_phs_true: np.ndarray,
                   selected_phs_eval: np.ndarray,
                   save_fname: str = None,
                   show_fig: bool = True):

    n = selected_diffs.shape[0]

    plt.viridis()

    f, ax = plt.subplots(7, n, figsize=(n * 4, 15))
    plt.gcf().text(0.02, 0.85, "Input", fontsize=20)
    plt.gcf().text(0.02, 0.72, "True I", fontsize=20)
    plt.gcf().text(0.02, 0.6, "Predicted I", fontsize=20)
    plt.gcf().text(0.02, 0.5, "Difference I", fontsize=20)
    plt.gcf().text(0.02, 0.4, "True Phi", fontsize=20)
    plt.gcf().text(0.02, 0.27, "Predicted Phi", fontsize=20)
    plt.gcf().text(0.02, 0.17, "Difference Phi", fontsize=20)

    for i in range(0, n):

        # display FT

        im = ax[0, i].imshow(np.log10(selected_diffs[i]))
        plt.colorbar(im, ax=ax[0, i], format='%.2f')
        ax[0, i].get_xaxis().set_visible(False)
        ax[0, i].get_yaxis().set_visible(False)

        # display predicted intens
        #im=ax[2,i].imshow(selected_amps_eval[i])
        #plt.colorbar(im, ax=ax[2,i], format='%.2f')
        #ax[2,i].get_xaxis().set_visible(False)
        #ax[2,i].get_yaxis().set_visible(False)

        # display original phase
        im = ax[4, i].imshow(selected_phs_true[i])
        plt.colorbar(im, ax=ax[4, i], format='%.2f')
        ax[4, i].get_xaxis().set_visible(False)
        ax[4, i].get_yaxis().set_visible(False)

        # display predicted phase
        im = ax[5, i].imshow(selected_phs_eval[i])
        plt.colorbar(im, ax=ax[5, i], format='%.2f')
        ax[5, i].get_xaxis().set_visible(False)
        ax[5, i].get_yaxis().set_visible(False)

        # Difference in phase
        im = ax[6, i].imshow(selected_phs_true[i] - selected_phs_eval[i])
        plt.colorbar(im, ax=ax[6, i], format='%.2f')
        ax[6, i].get_xaxis().set_visible(False)
        ax[6, i].get_yaxis().set_visible(False)

    if save_fname is not None:
        plt.savefig(save_fname)
    if show_fig:
        plt.show()
    else:
        plt.close()
