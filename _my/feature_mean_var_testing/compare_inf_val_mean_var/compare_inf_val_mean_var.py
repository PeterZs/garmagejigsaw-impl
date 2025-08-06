import os
import numpy as np
import matplotlib.pyplot as plt
import torch




def vis_once_hist(data1, data2, label1, label2, xlabel=None, ylabel=None, bins=30, alpha=0.5):

    # 可视化
    plt.figure(figsize=(8, 5))
    plt.hist(data1, bins=bins, alpha=alpha, label=label1, color='blue', edgecolor='black')
    plt.hist(data2, bins=bins, alpha=alpha, label=label2, color='orange', edgecolor='black')

    plt.legend()
    plt.title("")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"hist_{label1}_{label2}.png"))
    plt.show()


def vis_individual_lines(data1, data2, label1, label2, xlabel=None, ylabel=None):
    os.makedirs(save_dir, exist_ok=True)

    x = list(range(len(data1)))

    plt.figure(figsize=(10, 5))
    plt.plot(x, data1, label=label1, color='blue')
    plt.plot(x, data2, label=label2, color='orange')

    plt.legend()
    plt.title(f"Line Plot of {label1} and {label2}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"lines_{label1}_{label2}.png"))
    plt.show()


save_dir = "_my/feature_mean_var_testing/compare_inf_val_mean_var"


if __name__ == '__main__':
    inf_mean = np.load("_my/feature_mean_var_testing/cal_infset_mean_var/result/mean.npy")
    inf_var = np.load("_my/feature_mean_var_testing/cal_infset_mean_var/result/var.npy")
    val_mean = np.load("_my/feature_mean_var_testing/cal_valset_mean_var/result/mean.npy")
    val_var = np.load("_my/feature_mean_var_testing/cal_valset_mean_var/result/var.npy")
    running_mean = np.load("_my/feature_mean_var_testing/get_running_mean_var/result/mean.npy")
    running_var = np.load("_my/feature_mean_var_testing/get_running_mean_var/result/var.npy")
    a=1
    vis_once_hist(inf_mean, running_mean, "inf_mean", "running_mean", xlabel="mean", ylabel="count")
    vis_once_hist(inf_var, running_var, "inf_var", "running_var", xlabel="var", ylabel="count")
    vis_individual_lines(inf_mean, running_mean, "inf_mean", "running_mean", xlabel="feature", ylabel="mean")
    vis_individual_lines(inf_var, running_var, "inf_var", "running_var", xlabel="feature", ylabel="var")
    vis_individual_lines(np.sort(inf_mean), np.sort(running_mean), "inf_mean_sorted", "running_mean_sorted", xlabel="feature", ylabel="mean")
    vis_individual_lines(np.sort(inf_var), np.sort(running_var), "inf_var_sorted", "running_var_sorted", xlabel="feature", ylabel="var")