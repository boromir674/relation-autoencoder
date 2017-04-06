import os
import settings


class SingleLabelClusterEvaluation:
    """
    Responsible for executing the evaluations tasks (assesing performance against goldstandard, metrics computation/printing)
    for a spesific dataset split ('train', 'valid', 'test')
    """
    def __init__(self, split_goldstandard, split_label):
        """
        :param split_goldstandard: int => list of strings; maps example IDs to a list of given goldstandard labels (semantic relations). A list of more than one elements implies that there can be multiple correct/true labels (semantic relations) for the given example. This class however is concerned only by the first element of each list,
        :type split_goldstandard: dict
        :param split_label: the name of the split; one of {'train', 'valid', 'test'}
        :type split_label: str
        """
        assert split_label == settings.split_labels[0] or split_label == settings.split_labels[1] or split_label == settings.split_labels[2]
        self.split_label = split_label
        self.numberOfElements = 0
        self.induced_clusters = {}
        self.gold_clusters, self.assessableElemSet = SingleLabelClusterEvaluation._parse_first_relation_label(split_goldstandard)

    def feed_induced_clusters(self, response):
        """Reads predicted clusters\n
        Given the clusters induced for a dataset split computes the total number of clustered points and copies the input mapping ommiting keys pointing to empty sets (empty clusters)\n
        :param response: maps cluster ids (int) to sets of clustered example ids (ints)
        :type response: dict
        """
        self.numberOfElements = 0
        self.induced_clusters = {}
        for cluster_id, example_set in response.iteritems():
            if len(example_set) > 0:
                self.numberOfElements += len(example_set)
                self.induced_clusters[cluster_id] = set(example_set)

    def compute_metrics(self):
        """Compute f1, precision and recall scores.\n"""
        recB3 = self.b3_total_element_recall()
        precB3 = self.b3_total_element_precision()
        if recB3 == 0.0 and precB3 == 0.0:
            F1B3 = 0.0
        else:
            F1B3 = (2 * recB3 * precB3) / (recB3 + precB3)
        return F1B3, precB3, recB3

    def get_f1(self):
        recB3 = self.b3_total_element_recall()
        precB3 = self.b3_total_element_precision()
        if recB3 == 0.0 and precB3 == 0.0:
            return 0.0
        else:
            return (2 * recB3 * precB3) / (recB3 + precB3)

    def get_f_n(self, n):
        if n == 1:
            return self.get_f1()
        else:
            recB3 = self.b3_total_element_recall()
            precB3 = self.b3_total_element_precision()
            betasquare = n**2
            if recB3 == 0.0 and precB3 == 0.0:
                return 0.0
            else:
                return ((1 + betasquare) * recB3 * precB3) / ((betasquare * precB3) + recB3)

    def precision(self, retrieved_members, true_members):
        """Computes precision\n
        Calculates the precision metric given the retrieved elements (a single cluster's members) and the goldstandard elements of the class/cluster\n
        :param retrieved_members: the predicted members of a cluster
        :type retrieved_members: set
        :param true_members: the true members (goldstandard) of the cluster
        :type true_members: set
        :return: the precision as True Positives / (True Positives + False Positives)
        :rtype: float
        """
        return len(retrieved_members.intersection(true_members)) / float(len(retrieved_members.intersection(self.assessableElemSet)))

    def b3recall(self, retrieved_members, true_members):
        """Computes recall\n
        Calculates the recall metric given the retrieved elements (a single cluster's members) and the goldstandard elements of the class/cluster\n
        :param retrieved_members: the predicted members of a cluster
        :type retrieved_members: set
        :param true_members: the true members (goldstandard) of the cluster
        :type true_members: set
        :return: the precision as True Positives / (True Positives + False Negatives)
        :rtype: float
        """
        return len(retrieved_members.intersection(true_members)) / float(len(true_members))

    def b3_total_element_precision(self):
        totalPrecision = 0.0
        for cluster_id, predicted_members in self.induced_clusters.iteritems():
            for member_id in predicted_members:
                if member_id in self.assessableElemSet:
                    totalPrecision += self.precision(predicted_members, self._find_cluster(member_id, self.gold_clusters))
        return totalPrecision / float(len(self.assessableElemSet))

    def b3_total_cluster_precision(self):
        totalPrecision = 0.0
        for cluster_id, predicted_members in self.induced_clusters.iteritems():
            for member_id in predicted_members:
                if member_id in self.assessableElemSet:
                    totalPrecision += self.precision(predicted_members, self._find_cluster(member_id, self.gold_clusters)) \
                                      / (len(self.induced_clusters) * len(predicted_members))
        return totalPrecision

    def muc3_precision(self):
        numerator = 0.0
        denominator = 0.0
        for cluster_id, predicted_members in self.induced_clusters.iteritems():
            if len(predicted_members) > 0:
                numerator += self._len_assessable_response_cat(predicted_members) - self.overlap(predicted_members, self.gold_clusters)
                lenRespo = self._len_assessable_response_cat(predicted_members)
                if lenRespo != 0:
                    denominator += self._len_assessable_response_cat(predicted_members) - 1
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator

    def b3_total_element_recall(self):
        totalRecall = 0.0
        for cluster_id, predicted_members in self.induced_clusters.iteritems():
            for member_id in predicted_members:
                if member_id in self.assessableElemSet:
                    totalRecall += self.b3recall(predicted_members, self._find_cluster(member_id, self.gold_clusters))
        return totalRecall / len(self.assessableElemSet)

    def b3_total_cluster_recall(self):
        totalRecall = 0.0
        for cluster_id, predicted_members in self.induced_clusters.iteritems():
            for member_id in predicted_members:
                if member_id in self.assessableElemSet:
                    totalRecall += self.b3recall(predicted_members, self._find_cluster(member_id, self.gold_clusters)) / (len(self.induced_clusters) * len(predicted_members))
        return totalRecall

    def muc3_recall(self):
        numerator = 0.0
        denominator = 0.0
        for cluster_label, gold_members_ids in self.gold_clusters.iteritems():
            numerator += len(gold_members_ids) - self.overlap(gold_members_ids, self.induced_clusters)
            denominator += len(gold_members_ids) - 1
        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator

    def _len_assessable_response_cat(self, responesSet_c):
        length = 0
        for r in responesSet_c:
            if r in self.assessableElemSet:
                length += 1
        return length

    @staticmethod
    def _parse_relations(a_file, subset=None):
        """
        :param a_file: path to file with processed examples/sentences
        :type a_file: str
        :param subset: path to file with a dataset split. If specified the intersection with this set is taken
        :return:
        """
        if subset is None:
            with open(a_file, 'r') as ref_set:
                relations = {}
                for example_id, line in enumerate(ref_set):
                    relations[example_id] = line.split('\t')[-1].strip().split(' ')
            return relations
        else:
            with open(a_file, 'r') as ref_set:
                with open(subset, 'r') as split:
                    split_set = {}
                    for line in split:
                        if line not in split_set:
                            split_set[line] = 1
                relations = {}
                for example_id, line in enumerate(ref_set):
                    if line in split:
                        relations[example_id] = line.split('\t')[-1].strip().split(' ')
                    else:
                        relation[example_id] = ['']
            return relations

    @staticmethod
    def _parse_first_relation_label(relations):
        """
        Given the goldstandard id2list_of_labels dict, constructs the goldstandard dictionary mapping semantic relations (str) to set of example IDs (ints) and a set of example IDs (ints) with a goldstandard value found in the input goldstandard relations.\n
        :param relations: int => list of strings; maps example IDs to a list of given goldstandard labels (semantic relations). A list of more than one element implies that there can be multiple correct/true labels (semantic relations) for the given example. This class, however, is concerned only with the first element of each list.
        :type relations: dict
        :return: the goldstandard relation2ids dict and the set of labeled example IDs
        :rtype (dict, set)
        """
        gold_relation2ids = {}  # semantic_relation => set(example IDs), str => set() of ints
        labeled_examples_ids = set()  # a set of example IDs (ints) that have a goldstandard value as label
        for example_id, label_list in relations.iteritems():
            first_label = label_list[0]
            if first_label != '':
                labeled_examples_ids.add(example_id)
                if first_label in gold_relation2ids:
                    gold_relation2ids[first_label].add(example_id)
                else:
                    gold_relation2ids[first_label] = {example_id}
        return gold_relation2ids, labeled_examples_ids

    @staticmethod
    def _find_cluster(a, setsDictionary):
        for c in setsDictionary:
            if a in setsDictionary[c]:
                return setsDictionary[c]

    @staticmethod
    def overlap(reference_members, predicted_clusters):
        numberIntersections = 0
        for predicted_members in predicted_clusters.itervalues():
            if len(reference_members.intersection(predicted_members)) > 0:
                numberIntersections += 1
        return numberIntersections


def construct_split_evaluator(split_goldstandard, split_label):
    """
    :param split_goldstandard: int => list of strings; maps example IDs to a list of given goldstandard labels (semantic relations). A list of more than one elements implies that there can be multiple correct/true labels (semantic relations) for the given example. This class however is concerned only by the first element of each list,
    :type split_goldstandard: dict
    :param split_label: the name of the split; one of {'train', 'valid', 'test'}
    :type split_label: str
    :return: an object able to calculate metrics, given input queries by evaluating against the given goldstandard truth
    :rtype: SingleLabelClusterEvaluation
    """
    assert split_label == 'train' or split_label == 'valid' or split_label == 'test'
    return SingleLabelClusterEvaluation(split_goldstandard, split_label)


def construct_split_evaluator_from_file(goldstandard_file, split_label, subset=None):
    """
    :param goldstandard_file: path to file with processed examples/sentences
    :type goldstandard_file: str
    :param split_label: the name of the split; one of {'train', 'valid', 'test'}
    :type split_label: str
    :param subset: path to file with a dataset split. If specified the intersection with this set is taken
    :return: an object able to evaluate metrics, on a selected dataset split, given the goldstandard truth
    :rtype: SingleLabelClusterEvaluation
    """
    assert split_label == 'train' or split_label == 'valid' or split_label == 'test'
    if subset is None:
        with open(goldstandard_file, 'r') as ref_set:
            relations = {}
            for example_id, line in enumerate(ref_set):
                relations[example_id] = line.split('\t')[-1].strip().split(' ')
    else:
        with open(goldstandard_file, 'r') as ref_set:
            with open(subset, 'r') as split:
                split_set = {}
                for line in split:
                    if line not in split_set:
                        split_set[line] = 1
            relations = {}
            for example_id, line in enumerate(ref_set):
                if line in split:
                    relations[example_id] = line.split('\t')[-1].strip().split(' ')
                else:
                    relation[example_id] = ['']
    return SingleLabelClusterEvaluation(relations, split_label)
