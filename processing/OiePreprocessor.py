import os
import time
import argparse
import cPickle as pickle
from definitions import OieFeatures
from definitions.OieExample import OieExample


class FeatureLexicon:
    """
    A wrapper around various dictionaries storing the mined data. It holds 5 dictionaries in total. Two of them store
    mappings\n
    - str => int\n
    - int => str\n
    about all features extracted. An update in these dicts causes an update in the dict holding frequencies, which maps\n
    - str => float\n
    The final two dicts contain mappings\n
    - str => int
    - int => str\n
    about features which trigger with frequency that exceeds the given threshold.\n
    All the string dictionaries keys are of the form 'featDef#value' (i.e. 'posPatternPath#JJ_VV_NN', 'bigrams#e1_t2')
    """
    def __init__(self):
        self.nextId = 0  # var pointing to the next available id number to be used when inserting new stuff
        self.id2Str = {}  # map:   int => str
        self.str2Id = {}  # map:      str => int
        self.id2freq = {}  # map:  int => float. Gets updated only when 'get_or_add' ins invoked not 'get_or_add_pruned'
        self.nextIdPruned = 0  # pointer
        self.id2StrPruned = {}  # map: int => str
        self.str2IdPruned = {}  # map: str => int

    def get_or_add(self, s):
        """
        Returns the numerical ID of the input string mapped by the 'str2Id' dictionary and increments its frequency by 1.
            If the input string is not present, it inserts it in the 'str2Id' dict, sets its frequency to 1
            and returns its new numerical ID.
        :param s: string to search for. eg s = posPatternPath#JJ_VV_NN
        :return: the id of the input string as an integer
        """
        if s not in self.str2Id:
            self.id2Str[self.nextId] = s
            self.str2Id[s] = self.nextId
            self.id2freq[self.nextId] = 1
            self.nextId += 1
        else:
            self.id2freq[self.str2Id[s]] += 1
        return self.str2Id[s]

    def get_or_add_pruned(self, s):
        """
        Returns the numerical ID of the input string mapped by the 'str2IdPruned' dictionary.
            If the input string is not present, it inserts it in the 'str2IdPruned' dict and returns its new
            numerical ID. There is no frequency update here.
        :param s: string to search for belonging to the pruned ones, eg posPatternPath#NN_VV_ADJ_VBP
        :return: the id of the input string as an integer
        """
        if s not in self.str2IdPruned:
            self.id2StrPruned[self.nextIdPruned] = s
            self.str2IdPruned[s] = self.nextIdPruned
            self.nextIdPruned += 1
        return self.str2IdPruned[s]

    def get_id(self, a_string):
        """
        :param a_string: a feature such as 'bigrams#e1_t1'
        :return: the numerical ID from the str2Id dict
        """
        if a_string not in self.str2Id:
            return None
        return self.str2Id[a_string]

    def get_str(self, idx):
        """
        Returns the feature corresponding to the input numerical ID, as a string, eg 'bigrams#e1_t1'
        :param idx: a numerical ID
        :return: the feature corresponding to the input ID, mapped by sth id2Str dict
        """
        if idx not in self.id2Str:
            return None
        else:
            return self.id2Str[idx]

    def get_str_pruned(self, idx):
        """
        Returns the feature corresponding to the input numerical ID, only if the frequency of the feature triggering
            has passed a given threshold (if the key is found in the id2StrPruned dict). Returns None if not found.
        :param idx: a numerical ID
        :return: the feature function name concatenated with '#' and the string value of it (i.e. 'bigrams#e1_t1', 'arg1_lower#java')
        """
        if idx not in self.id2StrPruned:
            return None
        else:
            return self.id2StrPruned[idx]

    def get_freq(self, idx):
        """
        Returns the number of times the feature, corresponding to the input ID, has occured.
        :param idx: a numerical ID
        :return: the frequency of the input ID's feature
        """
        if idx not in self.id2freq:
            return None
        return self.id2freq[idx]

    def get_feature_space_dimensionality(self):
        """
        Returns the number of features that have passed the thresholding\n
        :return: the number of (unique) entries in the id2strPruned dict
        """
        return self.nextIdPruned


def build_feature_lexicon(raw_features, feature_extractors, lexicon):
    # invokes internally get_or_add building the str2Id, id2Str, id2freq dicts since expand parameter is True
    print 'Building feature lexicon...'
    for ex_f in raw_features:
        get_features(lexicon, feature_extractors, [ex_f[1], ex_f[4], ex_f[5], ex_f[7], ex_f[8], ex_f[6]], ex_f[2], ex_f[3], expand=True)
    print '  Lexicon now has {} unique entries'.format(lexicon.nextId)


def get_features(lexicon, feature_extractors, info, arg1=None, arg2=None, expand=False):
    """
    Returns a list of the numerical IDs of the features extracted from the input information. Input information
        represents a single sentence in the mined dataset.
    :type lexicon: FeatureLexicon
    :param feature_extractors: a list of feature extraction functions as the ones defined in OieFeatures.py eg [trigger,
        entityTypes, arg1_lower, arg2_lower, bow_clean, entity1Type, entity2Type, lexicalPattern, posPatternPath]
    :param info: a list containing information of the input datapoint\n
        Example\n
        parsing  : info[0] = '<-poss<-production->prep->for->pobj->'\n
        entities : info[1] = 'JOBTITLE-JOBTITLE'\n
        trig     : info[2] = 'TRIGGER:review|performance'\n
        sentence : info[3] = 'Supervised learning us a subset of learning methods'\n
        pos      : info[4] = 'DT NNP NNP , VBD IN DT NNP JJ NN NN NNP , VBZ JJ NN NN IN CD NNS IN DT NN NN IN DT NN .'\n
        docPath  : info[5] = './2000/01/01/1165031.xml'
    :param arg1: entity1 string, eg 'Java'
    :param arg2: entity2 string, eg 'C++'
    :type expand: Boolean flag controlling whether str2Id, id2Str and id2freq dictionaries should be expanded as new
        entries appear. If false it is assumed that inner dicts are already maximally populated.
    :return: the list of feature IDs
    """
    feats = []
    for f in feature_extractors:
        res = f(info, arg1, arg2)
        if res is not None:
            for feat_el in generate_feature_element(res):
                _load_features(lexicon, f.__name__ + "#" + feat_el, feats, expand=expand)
    return feats


def get_thresholded_features(lexicon, feature_extractors, info, arg1, arg2, threshold, expand=False):
    """
    Returns a list of the numerical IDs of the features extracted from the input information which frequency value
        exceed the given threshold. Input information represents a single sentence in the mined dataset.
    :type lexicon: FeatureLexicon
    :param feature_extractors: a list of feature exraction functions as the ones defined in OieFeatures.py eg [trigger,
        entityTypes, arg1_lower, arg2_lower, bow_clean, entity1Type, entity2Type, lexicalPattern, posPatternPath]
    :param info: a list containing information of the input datapoint\n
        Example\n
        - parsing  : l[0] = '<-poss<-production->prep->for->pobj->'\n
        - entities : l[1] = 'JOBTITLE-JOBTITLE'\n
        - trig     : l[2] = 'TRIGGER:review|performance'\n
        - sentence : l[3] = 'Supervised learning us a subset of learning methods'\n
        - pos      : l[4] = 'DT NNP NNP , VBD IN DT NNP JJ NN NN NNP , VBZ JJ NN NN IN CD NNS IN DT NN NN IN DT NN .'\n
        - docPath  : l[5] = './2000/01/01/1165031.xml'
    :param arg1: entity1 string, eg 'Java'
    :param arg2: entity2 string, eg 'C++'
    :param expand: flag controlling whether str2IdPruned, id2StrPruned dictionaries should be expanded as new
        entries appear. If false it is assumed that inner dicts are already maximally populated.
    :type expand: bool
    :param threshold: integer to cut-off low frequency feature strings, such as i.e. infrequent bigrams of the form [bigrams#e1_t1, bigrams#e1_t2, .., posPatternPath#JJ_VV_NN]
    :return: the list of feature IDs
    """
    feats = []
    for f in feature_extractors:
        res = f(info, arg1, arg2)
        if res is not None:
            for feat_el in generate_feature_element(res):
                _load_thresholded_features(lexicon, f.__name__ + "#" + feat_el, feats, threshold, expand=expand)
    return feats


def generate_feature_element(extractor_output):
    if type(extractor_output) == list:
        for _ in extractor_output:
            yield _
    else:
        yield extractor_output


def _load_features(lexicon, feat_str_id, feats, expand=False):
    if expand:
        feats.append(lexicon.get_or_add(feat_str_id))
    else:
        feat_id = lexicon.get_id(feat_str_id)
        if feat_id is not None:
            feats.append(feat_id)


def _load_thresholded_features(lexicon, feat_str_id, feats, thres, expand=False):
    if expand:
        if lexicon.id2freq[lexicon.get_id(feat_str_id)] > thres:
            feats.append(lexicon.get_or_add_pruned(feat_str_id))
    else:
        feat_id = lexicon.get_id(feat_str_id)
        if feat_id is not None:
            if lexicon.id2freq[feat_id] > thres:
                feats.append(lexicon.get_or_add_pruned(feat_str_id))


def read_examples(file_name):
    """
    Reads the input tab-separated (\\\\t) file and returns the parsed data as a list of lists of strings. Each line, of the file to read, corresponds to a datapoint and has as many entries as the number of elements of the list returned by definitions.OieFeatures.getBasicCleanFeatures plus one.
    Raises and IOError if a line found in the input file does not have 9 elements. The returned lists are of the form:\n
    ['counter_index', 'entry_1', 'entry_2', .., 'entry_9']\n
    A sample file with the required format is '../data-sample.txt'.\n
    :param file_name: a file path to read from
    :type file_name: str
    :return: of lists of strings. Each inner list has as first element a counter 0..N followed by the entries found in a line
        returned by definitions.OieFeatures.getBasicCleanFeatures corresponding to the ones in the input file
    :rtype: list
    """
    start = time.time()
    print 'Reading examples from tab separated file...'
    count = 0
    i = 0
    with open(file_name, 'r') as fp:
        relation_examples = []
        for i, line in enumerate(fp):
            line.strip()
            if len(line) == 0 or len(line.split()) == 0:
                raise IOError
            else:
                fields = line.split('\t')
                assert len(fields) == 9, "a problem with the file format (# fields is wrong) len is " + str(len(fields)) + "instead of 9"
                relation_examples.append([str(count)] + fields)
                count += 1
    print '  File contained {} lines'.format(i + 1)
    print '  Datapoints with valid features encoded: {}'.format(count)
    print '  Done in {:.2f} sec'.format(time.time() - start)
    return relation_examples


def load_features(raw_features_struct, lexicon, examples_list, labels_dict, threshold):
    """
    Encodes the input raw feature values into OieExample objects and appends to the input examples_list\n
    Reads relation labels_dict, from the input features if found, and updates the corresponding keys in input labels_dict with a list of tokens representing the label\n
    It also updates the "thresholded" 'str2IdPruned' and 'id2StrPruneddictionaries'
    .. seealso:: :funct:`read_examples`\nTypically, the input raw features data structure is generated by the above function.\n
    :param raw_features_struct: the input raw features data structure read from a tab separated file\n
    A list of lists with each inner list following the below decoder_type for 'getCleanFeatures':

    * feats[0] : counter
    * feats[1] : dependency parsing <-, ->, ...
    * feats[2] : entity 1 (eg java engineer)
    * feats[3] : entity 2 (eg software engineer)
    * feats[4] : entity-types-pair (eg JOBTITLE-JOBTITLE)
    * feats[5] : trigger (eg TRIGGER:is)
    * feats[6] : document path
    * feats[7] : whole sentence
    * feats[8] : sequence of pos tags between e1, e2 (exclusive)
    * feats[9] : given label for semantic relation/class
    * info = [feats[1], feats[4], feats[5], feats[7], feats[8], feats[6]]

    :type raw_features_struct: list of lists of strings
    :param lexicon: the dictionary "pruned" mappings are updated
    :type lexicon: FeatureLexicon
    :param examples_list: the list to populate with generated objects of type definitions.OieExample
    :type examples_list: list
    :param labels_dict: the dictionary to update the values with the read relation labels_dict (encoded as a list of tokens).
    :type labels_dict: dict example ID (int) => goldstandard label (list of tokens/strings)
    :param threshold: feature has to be found at least 'threshold' number of times
    :type threshold: int
    """
    start = time.clock()
    print "Creating training examples and putting into list structure..."
    index = 0
    for i, feats in enumerate(raw_features_struct):  # a list of lists of strings [[0, f1, f2, .., f9], [1, ..], .., [N, ..]]
        feat_ids = get_thresholded_features(lexicon, feat_extractors,
                                            [feats[1], feats[4], feats[5], feats[7], feats[8], feats[6]], feats[2], feats[3], expand=True,
                                            threshold=threshold)
        example = OieExample(feats[2], feats[3], feat_ids, feats[5], relation=feats[9])
        labels_dict[index] = feats[-1].strip().split(' ')
        index += 1
        examples_list.append(example)
    print '  Unique thresholded feature keys: {}'.format(lexicon.nextIdPruned)
    print '  Done in {:.1f} sec'.format(time.clock() - start)


def pickle_objects(feat_extrs, feat_lex, dataset_splits, goldstandard_splits, a_file):
    """Pickles the input objects in the specified file.

    :param feat_extrs: feature extractors
    :type feat_extrs: list of callable objects
    :param feat_lex: indexed feature values extracted from mined sentences
    :type feat_lex: FeatureLexicon
    :param dataset_splits: the collection of sentences split into 'train', 'test', 'valid' sets. Maps splits to examples
    :type dataset_splits: dict; str {'train', 'test', 'valid'} => list (of instances of definitions.OieExample)
    :param goldstandard_splits: the true relation labels. Maps splits {'train', 'test', 'valid'} to example-label mappings
    :type goldstandard_splits: dict; str {'train', 'test', 'valid'} => dict (int => list). List holds the tokens (strings) representing the label for the example int ID
    :param a_file: the target file to pickle the objects to
    :type a_file: str
    """
    start = time.time()
    print 'Pickling feature extraction functions, feature lexicon, dataset_splits batch examples and goldstandard_splits labels...'
    assert type(feat_extrs) == list, 'Expected a list of callables as the 1st object to be pickled'
    for _ in feat_extrs:
        assert callable(_) is True, 'Element {} of 1st object is not callable'.format(_)
    assert isinstance(feat_lex, FeatureLexicon), "Expected an instance of FeatureLexicon as the 2nd object to be pickled. Got '{}' instead".format(type(feat_lex))
    assert type(dataset_splits) == dict, 'Expected a dict as the 3rd object to be pickled'
    for _ in dataset_splits:
        assert _ in ['train', 'test', 'valid'],  "The dict expected as the 3rd object to be pickled, has key '{}' not in ['train', 'test', 'valid']".format(_)
    assert type(goldstandard_splits) == dict, 'Expected a dict as the 4th object to be pickled'
    for _ in goldstandard_splits:
        assert _ in ['train', 'test', 'valid'],  "The dict expected as the 4th object to be pickled, has key '{}' not in ['train', 'test', 'valid']".format(_)
    with open(a_file, 'wb') as pkl_file:
        pickle.dump(feat_extrs, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(feat_lex, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataset_splits, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(goldstandard_splits, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    print '  Done in {:.2f} sec'.format(time.time() - start)


def unpickle_objects(a_file, verbose=False, debug=False):
    """
    Unpickles the input file and returns references to the retrieved objects. Objects are assumed to have been pickled in the below order:\n
    * list: its elements are callable objects representing feature extrating functions
    * FeatureLexicon: holds the 5 dictionaries (mapping IDs, features (strings) and triggering frequencies), built from the mined sentences
    * dict: has keys 'train', 'test', 'dev' each mapping to a list of instances of type definitions.OieExample
    * dict: has keys 'train', 'test', 'dev' each mapping to a dict mapping integers (IDs) to lists of tokens. Each list can have one or more string tokens representing the relation label\n
    :param a_file: file containing pickled objects
    :type a_file: str
    :param verbose: prints informative messages
    :type verbose: bool
    :param debug: if true prints the type of each object loaded
    :type debug: bool
    :return: references to the unpickled objects
    :rtype: list, FeatureLexicon, dict, dict
    """
    start = time.time()
    with open(a_file, 'rb') as pkl_file:
        if verbose:
            print "Opened pickled file '{}'".format(a_file)
        feature_extraction_functions = pickle.load(pkl_file)
        if debug:
            print "Loaded object of type '{}'".format(type(feature_extraction_functions).__name__)
        assert type(feature_extraction_functions) == list
        the_relation_lexicon = pickle.load(pkl_file)
        if debug:
            print "Loaded object of type '{}'".format(type(the_relation_lexicon).__name__)
        assert isinstance(the_relation_lexicon, FeatureLexicon), "Expected an instance of FeatureLexicon as the 2nd object to be pickled. Got '{}' instead".format(type(the_relation_lexicon))
        the_dataset = pickle.load(pkl_file)
        if debug:
            print "Loaded object of type '{}'".format(type(the_dataset).__name__)
        assert type(the_dataset) == dict
        the_goldstandard = pickle.load(pkl_file)
        if debug:
            print "Loaded object of type '{}'".format(type(the_goldstandard).__name__)
        assert type(the_goldstandard) == dict
    if verbose:
        print '  loaded feature extractors:', ', '.join(("'" + str(_.__name__) + "'" for _ in feature_extraction_functions))
        print '  loaded dataset with {} splits'.format(', '.join(("'" + _ + "'" for _ in the_dataset.iterkeys())))
        print 'Done in {:.2f} sec'.format(time.time() - start)
    return feature_extraction_functions, the_relation_lexicon, the_dataset, the_goldstandard


def get_cmd_arguments():
    myparser = argparse.ArgumentParser(description='Processes an Oie file and add its representations to a Python pickled file.')
    myparser.add_argument('input_file', metavar='input-file',  help='input file in the Yao format, like data-sample.txt')
    myparser.add_argument('pickled_dataset', metavar='pickled-dataset', help='pickle file to be used to store output (created if empty)')
    myparser.add_argument('--batch', metavar='batch-name', default="train", nargs="?", help="name used as a reference in the pickled file, default is 'train'")
    myparser.add_argument('--thres', metavar='threshold-value', default="0", nargs="?", type=int, help='minimum feature frequency')
    myparser.add_argument('--test-mode', action='store_true', help='used for test files. If true the feature space is not expanded, so that previously unseen features are not added to the dicts')
    return myparser.parse_args()


if __name__ == '__main__':
    t_start = time.time()
    args = get_cmd_arguments()

    # reads the tabbed separated file into a list of lists of strings, representing extracted features
    exs_raw_features = read_examples(args.input_file)

    feat_extractors = OieFeatures.getBasicCleanFeatures()  # list of callable feature extraction functions
    relation_lexicon = FeatureLexicon()
    dataset = {}  # dict mapping keys 'train', 'test', 'dev' to a list of OieExample instances

    # dict mapping each key 'train', 'test', 'dev' to a dictionary mapping int to a list of strings, representing goldstandard relation labels
    # each inner list contains the tokens that comprise the label (i.e. ['is-a']). Most are expected to have a single token.
    goldstandard = {}

    if os.path.exists(args.pickled_dataset):  # if found pickled objects, else pickle into new file
        feat_extractors, relation_lexicon, dataset, goldstandard = unpickle_objects(args.pickled_dataset)

    examples = []  # list of instances of definitions.OieExample
    relation_labels = {}  # dictionary mapping int to list of strings

    if args.batch in dataset:
        examples = dataset[args.batch]  # list of OieExamples for the 'batch_name' input split of the dataset
        # dict with the goldstandard labels (lists of token(s)) for the 'batch_name' input split of the dataset
        relation_labels = goldstandard[args.batch]
    else:
        # insert the input batch name as a key in the 'dataset' dict, mapping to an empty list (for now)
        dataset[args.batch] = examples
        # insert the input batch name as a key in the 'goldstandard' dict, mapping to an empty dict (for now)
        goldstandard[args.batch] = relation_labels

    # update statistics and mappings for given split
    build_feature_lexicon(exs_raw_features, feat_extractors, relation_lexicon)

    # update the dataset split and goldstandard mappings with the thresholded extractions
    load_features(exs_raw_features, relation_lexicon, examples, relation_labels, args.thres)

    pickle_objects(feat_extractors, relation_lexicon, dataset, goldstandard, args.pickled_dataset)
