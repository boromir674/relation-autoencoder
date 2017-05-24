# from matplotlib import pyplot as pl
# from matplotlib.figure import *
import numpy as np
from matplotlib import pyplot as pl
from matplotlib.ticker import MaxNLocator


class Plot(object):
    def __init__(self, series, labels, x_label, y_label, y_low=None, y_high=None):
        assert len(series) == len(labels)
        for l in labels:
            if l == '':
                assert len(labels) == 1
        self.series = series
        self.labels = labels
        self.x_label = x_label
        self.y_label = y_label
        self.y_low = y_low
        self.y_high = y_high


def parse_data_series(a_file, columns):
    if type(columns) != list:
        columns = [columns]
    data_series = []
    for _ in columns:
        data_series_series.append([])
    with open(a_file, 'r') as f:
        for line in f.readlines():
            for i, col in enumerate(columns):
                data_series[i].append(line.split(' ')[col-1])
    return data_series


def create_single_plot_figure_from_file(a_file, columns, labels, title, x_label, y_label, save_name='fig'):
    series = parse_data_series(a_file, columns)
    my_plot = Plot(series, labels, x_label, y_label)
    fig = create_figure(my_plot, title)
    fig.savefig(save_name)


def create_figure(plots, a_title, grid=(1, 1)):
    """
    Creates a Figure object with subplots according to the given Plots. Each Plot instance can contain time series for one or more variables\n
    :param plots: each object is used to construct a subplot
    :type plots: list
    :param grid: the formation of the grid where the subplots lie
    :type grid: tuple
    :param a_title: the title to be given to the Figure
    :type a_title: str
    :return: the Figure object
    :rtype: matplotlib.figure.Figure
    """
    if type(plots) != list:
        plots = [plots]
    assert len(plots) == grid[0] * grid[1]
    fig = pl.figure(1)
    for i in xrange(grid[1]):
        for j in xrange(grid[0]):
            ax = fig.add_subplot(grid[0] * grid[1], i+1, j+1)
            ax.set_ylim(bottom=plots[i+j].y_low, top=plots[i+j].y_high, auto=True)
            ax.set_xlabel(plots[i+j].x_label)
            ax.set_ylabel(plots[i+j].y_label)
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
            for single_var_series, var_label in zip(plots[i+j].series, plots[i+j].labels):
                t = range(1, len(single_var_series)+1)
                line, = ax.plot(t, single_var_series)
                line.set_label(var_label)
            if len(plots[i+j].labels) != 1 or plots[i+j].labels[0] is not None:
                ax.legend()
    fig.suptitle(a_title, fontsize=12)
    return fig


if __name__ == '__main__':
    s1 = [-100, -50, -30, -28, -26, -25]
    s2 = [-120, -80, -50, -30, -25, -22]
    s3 = [0.2, 0.4, 0.5, 0.55]
    s4 = [0.1, 0.5, 0.8, 0.85, 0.9]
    l1 = 'pre'
    l2 = 'rec'
    labelss = [l1]
    title = 'Test metrics figure'
    # m_plot = Plot([s1, s2], [l1, l2], y_low=0, y_high=1)
    p3 = Plot([s1], ['neg induce_relations error'], x_label='epoch', y_label='error', y_low=None, y_high=None)
    p2 = Plot([s3], [l1], x_label='epoch', y_label='score',)
    # p3 = Plot([s1], [None], x_label='epoch', y_label='neg error', y_low=None, y_high=None)
    fig1 = create_figure([p2], title, grid=(1, 1))
    fig1.savefig('fig_1')
    create_single_plot_figure_from_file()
