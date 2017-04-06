import numpy as np

metrics = ['f1', 'pre', 'rec']


def get_evaluation_splits():
    return ['train', 'valid', 'test']


class TrainProductsLoader:

    def __init__(self, capacity):
        self.capacity = capacity
        self.train_error_series = np.empty(capacity)
        self.metrics = dict(zip(get_evaluation_splits(), [dict(zip(metrics, [np.empty(capacity)] * len(metrics)))] * len(get_evaluation_splits())))
        self.ind1 = 0
        self.ind2 = dict(zip(get_evaluation_splits(), [0] * len(get_evaluation_splits())))

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

if __name__ == '__main__':
    pr = TrainProductsLoader(4)
    print str(pr.metrics)
