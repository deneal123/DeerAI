import contextlib
import math
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics.utils.metrics
from PIL import Image, ImageDraw, ImageFont
from PIL import __version__ as pil_version

from ultralytics.utils import LOGGER, TryExcept, ops, plt_settings, threaded


def plot_results(file="path/to/results.csv", dir="", segment=False, pose=False, classify=False, on_plot=None, subtitle=None):
    """
    Plot training results from a results CSV file. The function supports various types of data including segmentation,
    pose estimation, and classification. Plots are saved as 'results.png' in the directory where the CSV is located.

    Args:
        file (str, optional): Path to the CSV file containing the training results. Defaults to 'path/to/results.csv'.
        dir (str, optional): Directory where the CSV file is located if 'file' is not provided. Defaults to ''.
        segment (bool, optional): Flag to indicate if the data is for segmentation. Defaults to False.
        pose (bool, optional): Flag to indicate if the data is for pose estimation. Defaults to False.
        classify (bool, optional): Flag to indicate if the data is for classification. Defaults to False.
        on_plot (callable, optional): Callback function to be executed after plotting. Takes filename as an argument.
            Defaults to None.

    Example:
        ```python
        from ultralytics.utils.plotting import plot_results

        plot_results('path/to/results.csv', segment=True)
        ```
    """
    import pandas as pd
    from scipy.ndimage import gaussian_filter1d

    save_dir = Path(file).parent if file else Path(dir)
    if classify:
        fig, ax = plt.subplots(2, 2, figsize=(6, 6), tight_layout=False)
        if subtitle is not None:
           fig.suptitle(subtitle, fontsize=12)
        index = [1, 4, 2, 5]
        index_val = [8, 9]
    elif segment:
        fig, ax = plt.subplots(2, 8, figsize=(18, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15, 16, 7, 8, 11, 12]
        fig.title(title)
    elif pose:
        fig, ax = plt.subplots(2, 9, figsize=(21, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 18, 8, 9, 12, 13]
        fig.title(title)
    else:
        fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
        index = [1, 2, 3, 4, 5, 8, 9, 10, 6, 7]
        fig.title(title)
    ax = ax.ravel()
    files = list(save_dir.glob("results*.csv"))
    assert len(files), f"No results.csv files found in {save_dir.resolve()}, nothing to plot."
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            val_box_loss = []
            val_cls_loss = []
            for i, j in enumerate(index_val):
                if j == 8 and classify:
                    val_box_loss = data.values[:, j].astype("float")
                elif j == 9 and classify:
                    val_cls_loss = data.values[:, j].astype("float")
                y = data.values[:, j].astype("float")
            for i, j in enumerate(index):
                y = data.values[:, j].astype("float")
                # y[y == 0] = np.nan  # don't show zero values
                if i == 0 and classify:
                    ax[i].plot(x, val_box_loss, label='валидация', linewidth=1, color='g')  # actual results
                    ax[i].plot(x, gaussian_filter1d(val_box_loss, sigma=3), ":", label="сглаженный\n(валидация)", linewidth=1, color='k')  # smoothing line
                elif i == 2 and classify:
                    ax[i].plot(x, val_cls_loss, label='валидация', linewidth=1, color='g')  # actual results
                    ax[i].plot(x, gaussian_filter1d(val_cls_loss, sigma=3), ":", label="сглаженный\n(валидация)", linewidth=1, color='k')  # smoothing line
                ax[i].plot(x, y, label='тренировка', linewidth=1, color='b')  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ":", label="сглаженный\n(обучение)", linewidth=1, color='y')  # smoothing line
                # ax[i].set_title(s[j], fontsize=9)
                o = ['box_loss', 'precision', 'cls_loss', 'recall']
                ax[i].set_title(o[i], fontsize=9)
                t = ['Значение функции потерь', 'Значение метрики точность', 'Значение функции потерь', 'Значение метрики полнота']
                ax[i].set_ylabel(t[i], fontsize=9)
                z = ['Номер эпохи', 'Номер эпохи', 'Номер эпохи', 'Номер эпохи']
                ax[i].set_xlabel(z[i], fontsize=9)
                ax[i].legend(fontsize=8)
                ax[i].set_xticks(range(0, len(x), (len(x)//5)))
        except Exception as e:
            LOGGER.warning(f"WARNING: Plotting error for {f}: {e}")
    fname = save_dir / "results.png"
    fig.savefig(fname, dpi=500)
    plt.close()
    if on_plot:
        on_plot(fname)


plot_results(file="C:/Users/NightMare/PycharmProjects/DeerAI/deer_detect/yolov8n/results.csv",
             classify=True,
             subtitle=None)
