import theano
import numpy as np
import scipy.sparse as sp
from collections import Counter


class DatasetSplit(object):
    """Encapsulates all the information for a dataset split. Typically representing one of {'train', 'valid'/'dev', 'test'} dataset split.\n
    Wrapper class around:\n
    * array of e1 entity IDs (int) of shape (l,) found in this split's sentences\n
    * array of e2 entity IDs (int) of shape (l,) found in this split's sentences.\n
    * feature triggering binary sparce matrix of shape (l, d), encoding the features extracted from the split's sentences.\n
    """
    def __init__(self, arguments1, arguments2, arg_features):
        """
        :param arguments1: array of the e1 entity IDs in the dataset 
        :param arguments2: array of the e2 entity IDs in the dataset
        :param arg_features: compressed sparse binary array with feature triggering for the dataset. Rows indicate example/sentence, columns feature values
        """
        self.args1 = arguments1  # (l) (number_of_examples)
        self.args2 = arguments2  # (l) (number_of_examples)
        self.xFeats = arg_features  # (l, h) (number_of_examples x number_of_features_passing_threshold])


class DatasetManager(object):
    """
    A wrapper around the autoencoder input dataset. Given a 'train' split and optionally 'valid' and 'test' splits holding instances of OieExamples:\n
    * creates ID => entity and entity => ID mappings\n
    * creates cumulative distribution of powered entity frequencies for sampling from\n
    * encodes the splits into lower level DatasetSplit wrappers
    """
    def __init__(self, oie_dataset, feature_lex, rng, neg_sampling_distr_power=0.75, verbose=False):
        """
        :param oie_dataset: a dictionary mapping keys 'train', 'test', 'valid' to lists of instances of OieExample
        :type oie_dataset: dict
        :param feature_lex: the feature lexicon pickled. Constructed from a tab separated file
        :type feature_lex: processing.OiePreprocessor.FeatureLexicon
        :param rng: a random number generator
        :type rng: np.random.RandomState
        :type rng: numpy.random.RandomState
        :param neg_sampling_distr_power: the power degree of the sampling distribution for negative sampling. The entities frequencies are powered to this number before computing the cumulative distribution for the sampling purposes. Default: 0.75
        """
        if 'train' not in oie_dataset:
            raise Exception("Dataset manager requires that the provided dataset contains a 'train' split.")
        self.negSamplingDistrPower = neg_sampling_distr_power
        self.rng = rng
        self.featureLex = feature_lex  # feature mappings: id2Str, str2Id
        self.split = {}
        entity_freqs = Counter([arg for arg in generate_args(oie_dataset)])  # str => freq
        if verbose:
            print '  feature space size: {}\n  number of unique entities: {}'.format(self.get_dimensionality(), len(entity_freqs))
        self.id2Arg, self.arg2Id = self._index_elements(entity_freqs)  # entity mappings
        norm1 = float(sum((_ for _ in self._generate_frequencies(entity_freqs))))
        self.negSamplingDistr = map(lambda x: x / norm1, (_ for _ in self._generate_frequencies(entity_freqs)))
        self.negSamplingCum = np.cumsum(self.negSamplingDistr)  # cumulative distribution (array)
        self.split['train'] = self._wrap_examples(oie_dataset, 'train')
        if verbose:
            print "  initialized 'train' split with {} number of examples".format(len(self.split['train'].args1))
        if 'valid' in oie_dataset:
            self.split['valid'] = self._wrap_examples(oie_dataset, 'valid')
            if verbose:
                print "  initialized 'valid' split with {} number of examples".format(len(self.split['valid'].args1))
        if 'test' in oie_dataset:
            self.split['test'] = self._wrap_examples(oie_dataset, 'test')
            if verbose:
                print "  initialized 'test' split with {} number of examples".format(len(self.split['test'].args1))

    def _wrap_examples(self, oie_examples, split):
        """
        Returns a data structure (DatasetManager) with arrays of the entities e1, e2 IDs for the input split examples, the feature triggering binary sparse array and arrays for e1, e2 with samples taken from the entities powered frequencies distribution.
        :param oie_examples: a list of instances of OieExample serving as the split under consideration
        :return: a DatasetSplit as a wrapper class around the input split.
        """
        l = len(oie_examples[split])  # number of examples
        # arrays of entity IDs; two per example
        args1 = np.zeros(l, dtype=np.int32)
        args2 = np.zeros(l, dtype=np.int32)
        # binary sparse matrix holding feature triggering. It is of shape [num_of_examples x num_of_extracted_features_passing_threshold]
        x_feats_dok = sp.dok_matrix((l, self.featureLex.get_feature_space_dimensionality()), dtype=theano.config.floatX)

        for i, oieEx in enumerate(oie_examples[split]):
            args1[i] = self.arg2Id[oieEx.arg1]
            args2[i] = self.arg2Id[oieEx.arg2]
            x_feats_dok[i, oieEx.features] = 1  # feature triggering binary matrix, oieEx.features (feat IDs) act as indices

        return DatasetSplit(args1, args2, sp.csr_matrix(x_feats_dok, dtype="float32"))

    def get_arg_voc_size(self):
        """Returns the number of (unique) entities encountered in the whole dataset (= union of 'train', 'test' and 'valid' splits)"""
        return len(self.arg2Id)

    def get_dimensionality(self):
        """Returns the number of unique featues extracted from the whole dataset"""
        return self.featureLex.get_feature_space_dimensionality()

    def get_example_feature(self, an_id, feature):
        """Returns the feature string (i.e. 'posPatternPath#POS_CD_NN_IN_DT') for the input example's ID (which is in the 'train' split), matching the input feature function string (i.e. 'posPatternPath'). Returns None if not found."""
        for e in self.split['train'].xFeats[an_id].nonzero()[1]:  # iterate through the indices of the features triggering of the examples in the 'train' split
            feat = self.featureLex.get_str_pruned(e)  # get feature string (i.e. 'arg1_lower#java'), if has passed thresholding
            if self.featureLex.get_str_pruned(e).find(feature) > -1:
                return feat
        return None

    def get_example_feature_valid(self, an_id, feature):
        """Returns the feature string (i.e. 'lexicalPattern#production_for') for the input example's ID (which is in the 'valid' split), matching the input feature function string (i.e. 'lexicalPattern'). Returns None if not found."""
        for e in self.split['valid'].xFeats[an_id].nonzero()[1]:
            feat = self.featureLex.get_str_pruned(e)
            if self.featureLex.get_str_pruned(e).find(feature) > -1:
                return feat
        return None

    def get_example_feature_test(self, an_id, feature):
        """Returns the feature string (i.e. "bow_clean#['george', 'balanchine', 'production', 'new', 'york', 'city', 'ballet']") for the input example's ID (which is in the 'test' split), matching the input feature function string (i.e. 'bow_clean'. Returns None if not found."""
        for e in self.split['test'].xFeats[an_id].nonzero()[1]:
            feat = self.featureLex.get_str_pruned(e)
            if self.featureLex.get_str_pruned(e).find(feature) > -1:
                return feat
        return None

    def get_neg_sampling_cum(self):
        """
        Returns the cumulative distribution of the powered entity frequencies.
        :return: a sorted array in ascending order. array[-1] = 1
        """
        return self.negSamplingCum

    def _generate_frequencies(self, frequencies):
        """Entity frequencies generator"""
        for id_x in xrange(len(self.id2Arg)):
            yield frequencies[self.id2Arg[id_x]] ** self.negSamplingDistrPower

    def generate_split_keys(self):
        for _ in ('train', 'valid', 'test'):
            if _ in self.split:
                yield _

    @staticmethod
    def _index_elements(elements):
        """
        Creates a two-way mapping between numerical IDs and strings representing entities (i.e. 'OOP'), given a dictionary having as keys the entities.\n
        :param elements: dictionary mapping entitiy strings (i.e. 'Java') to frequencies
        :return id2_elem: a dictionary mapping IDs (int) to entitiy strings
        :return elem2_id: a dictionary mapping entitiy strings to IDs (int).
        """
        idx = 0
        id2_elem = {}
        elem2_id = {}
        for x in elements:
            id2_elem[idx] = x
            elem2_id[x] = idx
            idx += 1
        return id2_elem, elem2_id


def generate_args(oie_dataset):
    """Generator of arguments e1 and e2\n
    Yields the arg1, arg2 attributes of each OieExample in each dataset split\n
    :param oie_dataset: the splits of the dataset encoded as instance OieExample's
    :type oie_dataset: dict; 'train', 'valid', 'test' => list of OieExample's
    :return: dataset e1 and e2 arguments
    :rtype: str
    """
    for split in ('train', 'valid', 'test'):
        if split in oie_dataset:
            for ex in oie_dataset[split]:
                yield ex.arg1
                yield ex.arg2
