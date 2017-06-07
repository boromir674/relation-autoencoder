import sys
import numpy as np
import settings as s
from evaluation.plot import *
import cPickle as pickle

metrics = ['f1', 'pre', 'rec']


class TrainProductsLoader:

    def __init__(self, capacity):
        self.capacity = capacity
        self.train_error_series = np.empty(capacity)
        self.metrics = dict(zip(s.split_labels, [dict(zip(metrics, [np.empty(capacity)] * len(metrics)))] * len(s.split_labels)))
        self.ind1 = 0
        self.ind2 = dict(zip(s.split_labels, [0] * len(s.split_labels)))

    def feed_train_error(self, epoch_error):
        assert self.ind1 < self.capacity, "Tried to record train error at iteration '{}', but maximum storing capacity is {}".format(self.ind1, self.capacity)
        self.train_error_series[self.ind1] = epoch_error
        self.ind1 += 1

    def feed_epoch_metrics(self, split, metrics_scores):
        assert split == 'train' or split == 'valid' or split == 'test'
        assert len(metrics_scores) == 3
        assert self.ind2[split] < self.capacity, "Tried to record retrieval metrics at iteration '{}', but maximum storing capacity is {}".format(self.ind2, self.capacity)
        for i, metric in enumerate(metrics):
            self.metrics[split][metric][self.ind2[split]] = metrics_scores[i]
        self.ind2[split] += 1

    def create_train_error_figure(self, name):
        fig = create_figure([Plot([self.train_error_series], labels=[None], x_label='epoch', y_label='error')], 'Error evolution')
        fig.savefig(s.plots_path + '/' + name)
    
    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # def create_prec_rec_graph(self, split, name):
    #     fig = create_figure([Plot()])
    #
    #     fig = pl.figure(1)
    #     for i in xrange(grid[1]):
    #         for j in xrange(grid[0]):
    #             ax = fig.add_subplot(grid[0] * grid[1], i + 1, j + 1)
    #             ax.set_ylim(bottom=plots[i + j].y_low, top=plots[i + j].y_high, auto=True)
    #             ax.set_xlabel(plots[i + j].x_label)
    #             ax.set_ylabel(plots[i + j].y_label)
    #             ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    #             for single_var_series, var_label in zip(plots[i + j].series, plots[i + j].labels):
    #                 t = range(1, len(single_var_series) + 1)
    #                 line, = ax.plot(t, single_var_series)
    #                 line.set_label(var_label)
    #             if len(plots[i + j].labels) != 1 or plots[i + j].labels[0] is not None:
    #                 ax.legend()
    #     fig.suptitle(a_title, fontsize=12)
    #     return fig


if __name__ == '__main__':
    pr = TrainProductsLoader(4)
    print str(pr.metrics)
