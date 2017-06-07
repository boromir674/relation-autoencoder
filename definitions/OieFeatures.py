import re
import os
import nltk
import string
from argparse import ArgumentParser

__author__ = 'diego'

parsing = 0  # info[parsing] : string with ->, <-, must be syntactic tree
entities = 1  # info[entities] : string like 'JOBTITLE-JOBTITLE'
trig = 2  # infor[trig] : string of the form 'TRIGGER:review|performance'
sentence = 3  # info[sentence] : string sentence
pos = 4  # info[pos] string of part-of-speech tags
docPath = 5


#  ======= Relation features =======

stopwords_list = nltk.corpus.stopwords.words('english')
_digits = re.compile('\d')


def bow(info, arg1, arg2):
    return info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()


def bow_clean(info, arg1, arg2):
    """
    Returns the BOW of between entities words (inclusive). Filters out stopwords and numbers
    :return: a list of words
    """
    bow = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    result = []
    tmp = []
    for word in bow:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    for word in tmp:
        if word not in stopwords_list and not _digits.search(word) and not word[0].isupper():
            result.append(word)
    return result


def before_arg1(info, arg1, arg2):
    """
    :return: one or two words, if possible, before entity_1
    """
    before = info[sentence][:info[sentence].find(arg1)]
    beforeSplit = before.lower().strip().split(' ')
    beforeSplit = [word for word in beforeSplit if word not in string.punctuation]
    # print beforeSplit
    if len(beforeSplit) > 1:
        return [beforeSplit[-2], beforeSplit[-1]]
    elif len(beforeSplit) == 1:
        if beforeSplit[0] != '':
            return [beforeSplit[-1]]
        else:
            return []
    else:
        return []


def after_arg2(info, arg1, arg2):
    """:return: 1 or 2 lowercase words, if possible, after entity_2"""
    after = info[sentence][info[sentence].rfind(arg2)+len(arg2):]
    afterSplit = after.lower().strip().split(' ')
    afterSplit = [word for word in afterSplit if word not in string.punctuation]
    if len(afterSplit) > 1:
        return [a for a in afterSplit[0: 2]]
    elif len(afterSplit) == 1:
        if afterSplit[0] != '':
            return [afterSplit[0]]
        else:
            return []
    else:
        return []


def bigrams(info, arg1, arg2):
    """
    :return: an array of lower cased bigrams, in the form [e1_t1, t1_t2, .., t5_e2]
    """
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_'+x[1] for x in zip(tmp, tmp[1:])]


def trigrams(info, arg1, arg2):
    """
    :return: an array of lower cased trigrams, in the form [e1_t1_t2, t1_t2_t3, .., t4_t5_e2]
    """
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_'+x[1]+'_'+x[2] for x in zip(tmp, tmp[1:], tmp[2:])]

def skiptrigrams(info, arg1, arg2):
    """
    :return: an array of lower cased skiptrigrams, in the form [e1_X_t2, t1_X_t3, .., t4_X_e2]
    """
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_X_'+x[2] for x in zip(tmp, tmp[1:], tmp[2:])]

def skipfourgrams(info, arg1, arg2):
    """
    :return: list of the form [e1_X_t1_t2, e1_t2_X_t3, t1_X_t3_t4, t1_t2_X_t4, ..., t7_t8_X_e2]
    """
    between = info[sentence][info[sentence].find(arg1):info[sentence].rfind(arg2)+len(arg2)].split()
    tmp = []
    for word in between:
        for pun in string.punctuation:
            word = word.strip(pun)
        if word != '':
            tmp.append(word.lower())
    return [x[0]+'_X_'+x[2] + '_' + x[3] for x in zip(tmp, tmp[1:], tmp[2:], tmp[3:])] +\
           [x[0]+'_'+x[1]+'_X_' + x[3] for x in zip(tmp, tmp[1:], tmp[2:], tmp[3:])]


def trigger(info, arg1, arg2):
    return info[trig].replace('TRIGGER:', '')


def entityTypes(info, arg1, arg2):
    return info[entities]


def entity1Type(info, arg1, arg2):
    return info[entities].split('-')[0]


def entity2Type(info, arg1, arg2):
    return info[entities].split('-')[1]


def arg1(info, arg1, arg2):
    return arg1


def arg1_lower(info, arg1, arg2):
    return arg1.lower()


def arg1unigrams(info, arg1, arg2):
    return arg1.lower().split()


def arg2(info, arg1, arg2):
    return arg2


def arg2_lower(info, arg1, arg2):
    return arg2.lower()


def arg2unigrams(info, arg1, arg2):
    return arg2.lower().split()


def lexicalPattern(info, arg1, arg2):
    """
    Strips the dependency path from arrows and tags retaining only the words
    :return: a string of the words in the syntactic path concatenated with '_'. The string is constructed from
        info[parsing] after replacing '->' and '<-' with ' ', keeping every odd number word and joining them with '_'
    """
    p = info[parsing].replace('->', ' ').replace('<-', ' ').split()
    result = []
    for num, x in enumerate(p):
        if num % 2 != 0:
            result.append(x)
    return '_'.join(result)


def dependencyParsing(info, arg1, arg2):
    return info[parsing]


def rightDep(info, arg1, arg2):
    p = info[parsing].replace('->', ' -> ').replace('<-', ' <- ').split()
    return ''.join(p[:3])


def leftDep(info, arg1, arg2):
    p = info[parsing].replace('->', ' -> ').replace('<-', ' <- ').split()
    return ''.join(p[-3:])


def posPatternPath(info, arg1, arg2):
    """
    :return: a string with the pos tags between entities (exclusive) of the form JJ_NN_VV_ ...
    """
    words = info[sentence].split()
    postags = info[pos].split()
    assert len(postags) == len(words), 'error'
    a = []
    for w in xrange(len(words)):
        a.append((words[w], postags[w]))
    if a:
        beginList = [a.index(item) for item in a if item[0] == arg1.split()[-1]] # index of right most token of ent1
        endList = [a.index(item) for item in a if item[0] == arg2.split()[0]] # index of left most token of ent2
        if len(beginList) > 0 and len(endList) > 0:
            # posPattern = [item[1] for item in a if beginList[0] > a.index(item) > endList[0]]
            posPattern = []
            for num, item in enumerate(a):
                if beginList[0] < num < endList[0]:
                    posPattern.append(item[1])
            return '_'.join(posPattern)
        else:
            return ''
    else:
        return ''


def getBasicCleanFeatures():
    """
    Returns a list with items as in the example below:
        trigger as 'TRIGGER:review|performance'\n
        entityTypes as 'JOBTITLE-JOBTITLE'\n
        arg1_lower as 'software engineer'\n
        arg2_lower as 'java engineer'\n
        bow_clean as ['java', 'engineer', 'is', 'a', 'type', 'of', 'software', 'engineer']\n
        entiy1Type as 'JOBTITLE'\n
        entiy2Type as 'SKILL'\n
        lexicalPattern as 'with_music_by'\n
        posPatternPath as 'VBP_CD_NN_IN'
    :return: function declarations; list of callables
    :rtype: list
    """
    features = [trigger, entityTypes, arg1_lower, arg2_lower, bow_clean, entity1Type, entity2Type, lexicalPattern,
                posPatternPath]
    return features


def get_arguments():
    args_parser = ArgumentParser(description='Demonstrate feature extraction')
    args_parser.add_argument('-d', nargs='+', default='all', help="The features of which to demonstrate extraction. It can be one of 'clean', 'all' or a list of feature definitions. By default demonstrate 'all' of them")
    args_parser.add_argument('-n', nargs='?', default='1', type=int)
    arguments = args_parser.parse_args()
    if arguments.d == ['all'] or arguments.d == ['clean']:
        arguments.d = arguments.d[0]
    return arguments


def demo(feature_def, n_points):
    """
    :param feature_def: an iterable of callables: feature functions definitions (i.e. [bow, trigger, bigrams, posPatternPath])
    :param n_points: the number of sentences for which to demonstrate feature extraction
    """
    max_len = max([len(f.__name__) for f in feature_def])
    with open(os.path.dirname(os.path.realpath(__file__)) +'/' + '../data-sample.txt', 'r') as fi:
        counter = 0
        line = fi.readline().strip()
        while line != '' and counter < n_points:
            if len(line) == 0 or len(line.split()) == 0:
                raise IOError
            else:
                fields = line.split('\t')
                # assert len(fields) == 9, "a problem with the file format (# fields is wrong) len is " + str(len(fields)) + " instead of 9"
            print fields[6], '\n'
            for f in feature_def:
                res = f([fields[0], fields[3], fields[4], fields[6], fields[7], fields[5]], fields[1], fields[2])
                print f.__name__ + ' '*(max_len-len(f.__name__)), ':', res
            line = fi.readline().strip()
            counter += 1
            print '\n'


bag_of_feature_defs = (bow, bow_clean, before_arg1, after_arg2, bigrams, trigrams, skiptrigrams, skipfourgrams, trigger, entityTypes, entity1Type, entity2Type, arg1, arg1_lower, arg1unigrams, arg2, arg2_lower, arg2unigrams, lexicalPattern, dependencyParsing, rightDep, leftDep, posPatternPath)


if __name__ == '__main__':
    args = get_arguments()
    print str(args)
    if args.d == 'all':
        feats = bag_of_feature_defs
    elif args.d == 'clean':
        feats = getBasicCleanFeatures()
    else:
        feats = [globals()[_] for _ in args.d]
    demo(feats, args.n)
