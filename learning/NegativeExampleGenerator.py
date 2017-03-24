import numpy as np


class NegativeExampleGenerator(object):
    def __init__(self, rand, neg_sampling_cum):
        """
        :param rand: an instance of class numpy.random.RandomState
        :param neg_sampling_cum: an array, sorted in ascending order, holding the cumulative ditribution: neg_sampling_cum[-1] = 1
        """
        self._rand = rand
        self._negSamplingCum = neg_sampling_cum
        assert abs(self._negSamplingCum[-1] - 1) < 1.e-4, 'Negative example generator initialized with a cumulative distribution derived from a non-normalized one'

    def get_negative_samples(self, num_positive_entities, num_negative_samples):
        """
        Randomly samples from the entity frequency distribution and returns the entity IDs as a (s, l) shaped array (samples, num_entities)\n
        :param num_positive_entities: 'l' number of datapoints/entities (e1 or e2) to sample for; deteremines the number of columns in the returned array.
        :type num_positive_entities: int
        :param num_negative_samples: 's' number of samples to take per datapoint/entity (e1 or e2); determines the number of rows in the returned array
        :type num_negative_samples: int
        :return: a (s,l) shaped array with each column holding 's' sampled IDs per entity e1 or e2 for all 'l' entities
        :rtype: numpy.ndarray
        """
        return self._get_sample(num_positive_entities * num_negative_samples).reshape((num_negative_samples, num_positive_entities))

    def _get_sample(self, num_samples):
        """
        Samples accrording to the entities frequency distribution, represented by the cumulative distribution. The returned elements can serve as entity IDs\n
        :return: an array of sampled indices 'ind', with 0 <= ind <= len(self._negSamplingCum) - 1.
        :rtype: numpy.ndarray
        """
        return np.array(map(self._negSamplingCum.searchsorted, self._rand.uniform(0, self._negSamplingCum[-1], num_samples)), dtype=np.int32)
