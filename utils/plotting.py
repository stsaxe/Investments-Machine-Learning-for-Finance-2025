import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

import configuration


def plot_column(dataset: pd.DataFrame, column: str, title: str, y_label: str, output_path: str,
                date_column: str = configuration.date_column) -> None:
    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.plot(dataset[date_column], dataset[column])
    plt.title(title, fontsize=configuration.font_size_title)
    plt.xlabel('Date', fontsize=configuration.font_size)
    plt.ylabel(y_label, fontsize=configuration.font_size)
    plt.tight_layout()
    plt.ylim(bottom=dataset[column].min())
    plt.xlim(left=dataset[date_column].min())
    plt.savefig(output_path + title, dpi=configuration.dpi_store)
    plt.show()
    print("\n")


def plot_correlation_matrix(matrix: pd.DataFrame, title: str, output_path: str) -> None:
    plt.figure(figsize=(10, 8), dpi=configuration.dpi_display)
    sns.heatmap(matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.yticks(fontsize=configuration.font_size)
    plt.xticks(fontsize=configuration.font_size)
    plt.title(title, fontsize=configuration.font_size_title)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()
    print("\n")


def plot_loss_curve(loss_train: list[float], loss_val: list[float], title: str, output_path: str) -> None:
    assert len(loss_train) == len(loss_val)

    x = range(1, 1 + len(loss_train))

    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.plot(x, loss_train, color='blue', label='Training')
    plt.plot(x, loss_val, color='orange', label='Validation')
    plt.yscale("log")
    plt.title(title, fontsize=configuration.font_size_title)
    plt.xlabel("Epochs", fontsize=configuration.font_size)
    plt.ylabel("Loss", fontsize=configuration.font_size)
    plt.xlim(left=1)
    plt.legend(loc='upper right', fontsize=configuration.font_size)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()


def plot_prediction_indexed(preds: list, targets: list, dates: list, title: str, output_path: str) -> None:
    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.plot(dates, targets, color='blue', label='Targets')
    plt.plot(dates, preds, color='orange', label='Predictions')
    plt.legend(loc='upper left', fontsize=configuration.font_size)
    plt.title(title, fontsize=configuration.font_size_title)
    plt.xlabel("Date", fontsize=configuration.font_size)
    plt.ylabel("Index", fontsize=configuration.font_size)
    plt.xlim(left=dates[0])
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()


def plot_prediction_returns(preds: list, targets: list, dates: list, initial_values: list[float], look_ahead: int,
                            title: str,
                            output_path: str, ) -> None:
    assert len(initial_values) == look_ahead

    targets_idx = initial_values
    preds_idx = []

    for r in targets:
        targets_idx.append(targets_idx[-look_ahead] * (1 + r))

    for i, r in enumerate(preds):
        preds_idx.append(targets_idx[i] * (1 + r))

    targets_idx = targets_idx[look_ahead:]

    assert len(targets_idx) == len(preds_idx)
    assert len(targets_idx) == len(dates)

    plot_prediction_indexed(preds_idx, targets_idx, dates=dates, title=title, output_path=output_path)


def plot_feature_importance(labels: list[str], sizes: list[float], title: str, output_path: str) -> None:
    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.barh(labels[::-1], sizes[::-1])
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.xlim(left=0, right=110)
    plt.title(title, fontsize=configuration.font_size_title)
    plt.yticks(fontsize=configuration.font_size)
    plt.xticks(fontsize=configuration.font_size)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()


def plot_return_density(preds: list[float], targets: list[float], title: str, output_path: str) -> None:
    preds = [i * 100 for i in preds]
    targets = [i * 100 for i in targets]

    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.hist(targets, bins=15, histtype='step', color='blue', density=True, label='Targets')
    plt.hist(preds, bins=15, histtype='step', color='orange', density=True, label='Predictions')
    plt.title(title, fontsize=configuration.font_size_title)
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.legend(loc='upper right', fontsize=configuration.font_size)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()


def plot_scatter(x: list[float], y: list[float], xlabel: str, ylabel: str, title: str, output_path: str) -> None:
    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.scatter(x, y)
    plt.title(title, fontsize=configuration.font_size_title)
    plt.xlabel(xlabel, fontsize=configuration.font_size)
    plt.ylabel(ylabel, fontsize=configuration.font_size)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()
