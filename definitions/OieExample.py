class OieExample (object):
    """
    This is the basic wrapper class around a sentence datapoint. It holds information about features as numerical IDs and
    e1, e2 entities, the trigger, and possible (true) label as strings.
    """
    def __init__(self, arg1, arg2, features, trigger, relation=''):
        """
        :param arg1: e1 entity as a string, eg 'Java'
        :param arg2: entity2 as a string, eg 'Python'
        :param features: a list of numerical IDs of features. Obtained with get_thresholded_features or get_features of
            Preprocessor.py
        :param trigger: predicate as a string
        :param relation: the given/"correct" label/class
        :type relation: str
        """
        self.features = features
        self.arg1 = arg1
        self.arg2 = arg2
        self.relation = relation
        self.trigger = trigger
