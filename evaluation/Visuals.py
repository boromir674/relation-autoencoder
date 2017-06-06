import operator
from collections import Counter


class ModelPrinter(object):

    def __init__(self, a_trained_model):
        """
        :param a_trained_model:
        :type a_trained_model: learning.OieInduction.ReconstructInducer
        """
        self.model = a_trained_model

    def print_triggers(self, split, nb_elements):
        """
        Prints the clustered triggers, found in the specified dataset split
        :param split: {'train', 'valid', 'test'}
        :param nb_elements:
        """
        Visualizator.print_clusters(self.model.cluster[split], self.model.data, split, nb_elements)

    def print_labels(self, split, nb_elements):
        """
        Print groundtruth/goldstandard labels, found in the specified dataset split
        :param split: {'train', 'valid', 'test'}
        :param nb_elements:
        """
        Visualizator.print_clusters(self.model.cluster[split], self.model.data, split, nb_elements, goldstandard=self.model.goldStandard)


class Visualizator(object):

    @staticmethod
    def print_clusters(cluster_sets, dataset_manager, split, threshold, goldstandard=None):
        """Visualize clusters' examples triggers or true labels up to the specified maximum, from the input 'split' of the dataset.\n
        Prints the member triggers/true labels found in the input split of the dataset for each cluster induced. Maximal number of elements visualized pre cluster cannot exceed the given 'threshold' value.\n
        :param cluster_sets: mapping cluster IDs to sets of indices (int) pointing to examples
        :type cluster_sets: dict
        :param dataset_manager: a reference to the object acting as a wrapper around the dataset. Used to get the reference of the 'feature extractor' method corresponding to the given 'split' of the dataset.
        :type dataset_manager: learning.OieData.DatasetManager
        :param split: determines in which 'split' of the dataset to look for mined triggers
        :type split: str {'train', 'valid', 'test'}
        :param threshold: the maximum number of trigger-frequency pairs to visualize per cluster
        :type threshold: int
        :param goldstandard: the goldstandard relation labels: a dict that maps splits ('train', 'test', 'valid') to dictionaries (int => list) that encode example-to-label mappings [int => list of tokens (strings)]
        :type goldstandard: dict
        """
        if goldstandard is None:
            extractor = Visualizator._get_trigger_extractor(dataset_manager, split)
        else:
            extractor = Visualizator._get_label_extractor(goldstandard, split)
        for cl_id, cl_members in cluster_sets.items():
            print cl_id,
            trig_or_labels = [extractor(idx) for idx in cl_members]
            frequencies = Counter([idx for idx in trig_or_labels if idx is not None])
            Visualizator.print_clustered_triggers(frequencies, threshold)

    @staticmethod
    def print_clustered_triggers(triggers2freq, threshold):
        """Visualize triggers/labels for the examples in a single cluster\n
        Takes a dictionary (trigger => frequency) corresponding to a single cluster (with member triggers)
        and a threshold value and prints the most frequent triggers and their frequency. The number of visualized triggers cannot be more than the input threshold value\n
        Visualization format: (id_1, freq_1) (id_2, freq_2) ... (id_n, freq_n)
         where 'n' can maximally reach the input threshold\n
        :param triggers2freq: the cluster triggers
        :type triggers2freq: dict mapping str => int
        :param threshold: the maximum number of trigger-frequency pairs to print
        :type threshold: int
        """
        _ = sorted(triggers2freq.items(), key=operator.itemgetter(1), reverse=True)  # list of key, val tuples sorted in descending order by freq
        print ' '.join(map(str, _[:threshold]))

    @staticmethod
    def _get_trigger_extractor(dataset_manager, split):
        return Visualizator._get_trigger(split, dataset_manager.get_example_feature)

    @staticmethod
    def _get_label_extractor(goldstandard, split):
        return lambda x: goldstandard[split][x][0]

    @staticmethod
    def _get_trigger(split, extractor):
        return lambda x: Visualizator._remove_trigger_token(extractor(x, split, 'trigger'))

    @staticmethod
    def _remove_trigger_token(trigger):
        if trigger is None:
            return None
        else:
            return trigger.replace('trigger#', '')
