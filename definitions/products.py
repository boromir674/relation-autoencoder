import numpy as np
import settings as s
from evaluation.plot import *

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

    def create_train_error_figure(self):
        fig = create_figure([Plot([self.train_error_series], labels=[None], x_label='epoch', y_label='error')], 'Error evolution')
        fig.savefig('test_fig')

if __name__ == '__main__':
    pr = TrainProductsLoader(4)
    print str(pr.metrics)
