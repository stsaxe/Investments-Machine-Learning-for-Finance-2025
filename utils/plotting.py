import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import configuration


def plot_column(dataset: pd.DataFrame, column: str, title: str, y_label: str, output_path: str,
                date_column: str = configuration.date_column) -> None:
    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.plot(dataset[date_column], dataset[column])
    plt.title(title, fontsize=configuration.font_size_title)
    plt.xlabel('Date')
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.ylim(bottom=dataset[column].min())
    plt.xlim(left=dataset[date_column].min())
    plt.savefig(output_path + title, dpi=configuration.dpi_store)
    plt.show()
    print("\n")


def plot_correlation_matrix(matrix: pd.DataFrame, title: str, output_path: str) -> None:
    plt.figure(figsize=(10, 8), dpi=configuration.dpi_display)
    sns.heatmap(matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.title(title, fontsize=configuration.font_size_title)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()
    print("\n")


def plot_prediction_indexed(preds: list, targets: list, title: str, output_path: str) -> None:
    x = range(len(targets))

    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.plot(x, targets, color='blue', label='Targets')
    plt.plot(x, preds, color='orange', label='Predictions')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()


def plot_prediction_returns(preds: list, targets: list, title: str, look_ahead: int, output_path: str) -> None:
    chart_values = [100 for i in range(look_ahead)]
    chart_preds = [100 for i in range(look_ahead)]

    for r in targets:
        chart_values.append(chart_values[-look_ahead] * (1 + r))

    for i, p in enumerate(preds):
        chart_preds.append(chart_values[i] * (1 + p))

    chart_preds = chart_preds[look_ahead:]
    chart_values = chart_values[look_ahead:]

    x = range(len(targets))

    plt.figure(figsize=configuration.fig_size, dpi=configuration.dpi_display)
    plt.plot(x, chart_values, color='blue', label='Targets')
    plt.plot(x, chart_preds, color='orange', label='Predictions')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path + title, dpi=configuration.dpi_store * 2)
    plt.show()
