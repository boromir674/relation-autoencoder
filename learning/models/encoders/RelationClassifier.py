import theano
import numpy as np
from theano import sparse
import theano.tensor as T
import settings


class IndependentRelationClassifiers(object):
    """
    An implementation of a maxEnt classifier. Provides interface to compute classed probabilities and to label input features\n
    - W  : [features_space_dimensionality x number_of_classes] initialized uniform randomly in [settings.low, settings.high] range\n
    - Wb : [number_of_classes x 1] initialized with zeros
    """
    def __init__(self, rng, feature_dim, relation_num):
        """
        :type rng: instance of numpy.random.RandomState
        :param feature_dim: dimension of the binary feature space; number of distinct features extracted
        :param relation_num: number of classes; user defined number of semantic relation_labels/clusters to induce
        """
        self.d = feature_dim
        self.m = relation_num
        # print 'Sampling range for initialization of W weights: low:', settings.low, 'high:', settings.high
        # weights initialization; random for matrix W and zeros for vector Wb
        self.W = theano.shared(np.asarray(rng.uniform(low=settings.low, high=settings.high, size=(self.d, self.m)), dtype=theano.config.floatX), name='W', borrow=True)
        self.Wb = theano.shared(value=np.zeros(self.m, dtype=theano.config.floatX), name='Wb', borrow=True)
        self.params = [self.W, self.Wb]

    def comp_relation_probs(self, x_feats):
        """Function that holds the computations for the class probabilities\n
        :param x_feats: a sparse binary matrix variable of shape (number_of_examples, feature_space_dimensionality)
        :return: an array of shape (l, m) (number_of_examples, number_of_classes) with class probabilities per example/sentence
        """
        # l : examples batch size
        # d : dimensionality of the (binary) feature space
        relation_scores = sparse.dot(x_feats, self.W) + self.Wb   # [l, d] x [d, m] + [m] => [l, m]
        relation_probs = T.nnet.softmax(relation_scores)
        return relation_probs

    def comp_probs_and_labels(self, x_feats):
        """Holds the computations both for labeling the input features and calculating class probabilities\n
        Performs classification labeling (argmax) and computing probabilities of classes given the input feature matrix. Matrix represent examples.\n
        :param x_feats: a binary array of shape (l,d) = [number_of_examples x feature_space_dimensionality]
        :return: a (l,) shaped array of numerical labels and a (l, m) array of probabilities as a tuple
        """
        scores = sparse.dot(x_feats, self.W) + self.Wb  # [l, d] x [d, m] + [m] => [l, m]
        relation_probs = T.nnet.softmax(scores)
        labels = T.argmax(scores, axis=1)  # [l, m] => [l]
        return labels, relation_probs
