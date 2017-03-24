import os
import sys
import time
import theano
import argparse
import operator
import numpy as np
import theano.sparse
import cPickle as pickle
import theano.tensor as T
import settings as settings
from collections import Counter
from learning.OieData import DatasetSplit
from learning.OieData import DatasetManager
from learning.OieModel import OieModelFunctions
from processing.OiePreprocessor import FeatureLexicon
from evaluation.Visuals import Visualizator as printer
from processing.OiePreprocessor import unpickle_objects
from evaluation.OieEvaluation import SingleLabelClusterEvaluation
from learning.NegativeExampleGenerator import NegativeExampleGenerator


class ReconstructInducer(object):

    def __init__(self, data, gold_standard, rng, nb_epochs, learning_rate, batch_size, embed_size, nb_relations, nb_neg_samples, lambda1, lambda2,
                 optimization, model_name, decoder_model, external_embeddings, extended_regularizer,
                 frequent_eval, alpha):
        """
        :param data: the indexed input examples set. Contains 'train' split and optionally 'test' and 'valid' splits.
        :type data: learning.OieData.DatasetManager
        :param gold_standard: the goldstandard relation labels: a dict that maps splits ('train', 'test', 'valid') to dictionaries that encode example-to-label mappings [int => list of tokens (strings)]
        :type gold_standard: dict
        :param rng: a random number generator
        :type rng: numpy.random.RandomState
        :param nb_epochs: number of training iterations (epochs)
        :type nb_epochs: int
        :param learning_rate: controls the rate of the weight updating, 0 < rate :math:`\\leq` 1
        :type learning_rate: float
        :param batch_size: determines the number of considered datapoints for the computation of the train error
        :type batch_size: int
        :param embed_size: integer representing the dimensionality of entity embeddings created, r in notes
        :type embed_size: int
        :param nb_neg_samples: determines the number of samples to take per entity e1 and e2 for the 'Negative sampling' approximation
        :type nb_neg_samples: int
        :param lambda1: parameter :math:`\\lambda` :math:`\\geq` 0 for the L1-norm regularization
        :type lambda1: float
        :param lambda2: parameter :math:`\\lambda` :math:`\\geq` 0 for the L2-norm regularization
        :type lambda2: float
        :param optimization: optimization algorithm to use: {adagrad, sgd} for {AdaGrad, Stochastic Gradient Descend}
        :type optimization: str
        :param model_name: the name of the system/pipeline to use when saving to disc
        :type model_name: str
        :param decoder_model: decoder_model to deploy; implemented 'RESCAL', 'Selectional Preferences', 'Hybrid of RESCAL + SP'
        :type decoder_model: str {'rescal', 'sp', 'rescal+sp'}
        :param external_embeddings: if True then word2vec generated embeddings are used to initialize the entity embeddings.
        :type external_embeddings: bool
        :param extended_regularizer: if True uses entropy regularizer for the decoder's weights, in addition to the encoder's ones. If false regularizes only encoder's weights.
        :type extended_regularizer: bool
        :param frequent_eval: If true uses frequent evaluation... TODO
        :type frequent_eval: bool
        :param alpha: real number in range [0,1] used for scaling the entropy term in order to match the scale of the expected approximated softmax
        :type alpha: float
        """
        self.data = data
        self.goldStandard = gold_standard
        self.rand = rng
        self.nb_epochs = nb_epochs
        self.learningRate = learning_rate
        self.batch_size = batch_size
        self.embedSize = embed_size
        self.relationNum = nb_relations
        self.neg_sample_num = nb_neg_samples
        self.lambdaL1 = lambda1
        self.lambdaL2 = lambda2
        self.optimization = optimization
        self.modelName = model_name
        self.decoder = decoder_model
        self.extEmb = external_embeddings
        self.extendedReg = extended_regularizer
        self.frequentEval = frequent_eval
        self.alpha = alpha
        self.negativeSampler = NegativeExampleGenerator(rng, data.negSamplingCum)
        self.accumulator = []
        self.modelID = decoder_model + '_' + model_name + '_maxepoch' + str(nb_epochs) + '_lr' + str(learning_rate) + \
            '_embedsize' + str(embed_size) + '_l1' + str(lambda1) + '_l2' + str(lambda2) + '_opt' + str(optimization) + \
            '_rel_num' + str(self.relationNum) + '_batch' + str(batch_size) + '_negs' + str(self.neg_sample_num)
        self.modelFunc = OieModelFunctions(rng, embed_size, nb_relations, nb_neg_samples, batch_size, decoder_model, self.data, self.extendedReg, self.alpha, external_embeddings=self.extEmb)

    def compile_function(self):
        """ ReconstructInducer compiling capabilities\n
        Compiles the train symbolic function, based on the objective of minimizing the reconstruction error, for joint optimization of the autoencoder parameters.
        Gradients are computed on this symbolic function. It also compiles a labeling function for each of the dataset splits according to the self.data DatasetSplit instance.\n
        :return: Returns the compiled 'train_function' followed by 1-3 labeling functions in ordered as ['train', 'valid', 'test']
        :rtype: list
        """
        batch_index = T.lscalar()  # index to the starting index of a mini-batch
        ents_1 = T.ivector()  # (l,)
        ents_2 = T.ivector()  # (l,)
        neg1 = T.imatrix()  # (s,l)
        neg2 = T.imatrix()  # (s,l)
        xFeats = theano.sparse.csr_matrix(name='x', dtype='float32')  # (l, d)
        # ratio of batch size to the number of datapoints/sentences in the train set
        adjust = float(self.batch_size) / float(self.data.split['train'].args1.shape[0])

        # autoencoder negation of objective function computation expression
        cost = self.modelFunc.build_train_err_computation(xFeats, ents_1, ents_2, neg1, neg2) +\
            (self.lambdaL1 * self.modelFunc.L1 * adjust) + (self.lambdaL2 * self.modelFunc.L2 * adjust)

        optimizer = self._initialize_optimization_algorithm()  # responsilbe for updating weights/parameters
        updates = optimizer.update(self.learningRate, self.modelFunc.params, cost)

        train_split = make_shared(self.data.split['train'])

        print '  Compiling train function'
        # takes as input an index to the starting point of a mini-batch, and two vectors of the same length as mini-batch.
        # the vectors hold indices to the entities sampled for the 'negative sampling' approximation of the posterior

        trainModel = theano.function(inputs=[batch_index, neg1, neg2], outputs=cost, updates=updates, allow_input_downcast=True,
                                     givens={xFeats: train_split.xFeats[batch_index * self.batch_size: (batch_index + 1) * self.batch_size],
                                             ents_1: train_split.args1[batch_index * self.batch_size: (batch_index + 1) * self.batch_size],
                                             ents_2: train_split.args2[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]})
        # maximun likelihood prediction labeling computation expression
        prediction = self.modelFunc.build_label_computation(xFeats)  # tuple

        labeling_functions = []
        for key in self.data.generate_split_keys():
            print "  compiling labeling function for the '" + key + "' set"
            labeling_functions.append(theano.function(inputs=[batch_index], outputs=prediction, updates=[], givens={
                xFeats: make_shared(self.data.split[key]).xFeats[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]}))

        return [trainModel] + labeling_functions

    def learn(self, debug=False):

        print 'Compiling...'
        functions = self.compile_function()

        # compute number of minibatches for train, validation and test sets
        batch_reps = [self.data.split[_].args1.shape[0] / self.batch_size for _ in self.data.generate_split_keys()]
        evaluations = [SingleLabelClusterEvaluation(self.goldStandard[_], False) for _ in self.data.generate_split_keys()]

        print 'Training model on {} examples'.format(batch_reps[0] * self.batch_size)
        startTime = time.clock()
        doneLooping = False
        epoch = 0

        while (epoch < self.nb_epochs) and (not doneLooping):
            epochStartTime = time.clock()
            err = 0
            epoch += 1
            print '\nEPOCH', epoch
            negativeSamples1 = self.negativeSampler.get_negative_samples(self.data.split['train'].args1.shape[0], self.neg_sample_num)
            negativeSamples2 = self.negativeSampler.get_negative_samples(self.data.split['train'].args2.shape[0], self.neg_sample_num)

            for idx in xrange(batch_reps[0]):
                neg1 = negativeSamples1[:, idx * self.batch_size: (idx + 1) * self.batch_size]
                neg2 = negativeSamples2[:, idx * self.batch_size: (idx + 1) * self.batch_size]
                err += functions[0](idx, neg1, neg2)

                if self.frequentEval:
                    if self._mode(self.data) == 1:
                        print idx * self.batch_size, idx, '############################################################'
                        print self.get_clusters_size(functions[1], batch_reps[0]), '\n'
                    elif self._mode(self.data) == 2:
                        print idx * self.batch_size, idx, '############################################################'
                        for i, key in zip((1, 2), ('Validation', 'Test')):
                            cluster = self.get_clusters_sets(functions[i + 1], batch_reps[i])
                            evaluations[i].create_response(cluster)
                            evaluations[i].print_metrics(key)

            epochEndTime = time.clock()
            print 'Training error: {:.4f}'.format(err)
            print 'Epoch duration: {:.1f}s'.format(epochEndTime - epochStartTime)

            if self._mode(self.data) == 1:
                print 'Training Set'
                trainClusters = self.get_clusters_sets(functions[1], batch_reps[0])
                posteriorsTrain = [functions[1](i)[1] for i in xrange(batch_reps[0])]
                trainPosteriors = [item for sublist in posteriorsTrain for item in sublist]
                evaluations[0].create_response(trainClusters)

                if self.modelName != 'Test':
                    evaluations[0].print_metrics('Training')
                    # printer.print_clusters(trainClusters, self.data, 'train', settings.elems_to_visualize)
                if self.modelName == 'Test':
                    print 'Test print'
                    printer.print_clusters(trainClusters, self.data, 'train', settings.elems_to_visualize, goldstandard=self.goldStandard)
                    # self.getClustersWithFrequencies(trainClusters, self.data, settings.elems_to_visualize)
                else:
                    printer.print_clusters(trainClusters, self.data, 'train', settings.elems_to_visualize)
                    # getClustersWithFrequencies(trainClusters, self.data, settings.elems_to_visualize)

                if debug:
                    pickle_clustering(trainClusters, self.modelID+'_epoch'+str(epoch))
                    if epoch % 5 == 0 and epoch > 0:
                        pickle_posteriors(trainPosteriors, self.modelID+'_Posteriors_epoch'+str(epoch))

            if self._mode(self.data) == 2:
                validCluster = self.get_clusters_sets(functions[2], batch_reps[1])
                posteriorsValid = [functions[2](i)[1] for i in xrange(batch_reps[1])]
                validPosteriors = [item for sublist in posteriorsValid for item in sublist]
                evaluations[1].create_response(validCluster)
                evaluations[1].print_metrics('Validation')
                printer.print_clusters(validCluster, self.data, 'valid', settings.elems_to_visualize)
                # getClustersWithFrequenciesValid(validCluster, self.data, settings.elems_to_visualize)

                if debug:
                    pickle_clustering(validCluster, self.modelID+'_epoch'+str(epoch)+'_valid')
                    if epoch % 5 == 0 and epoch > 0:
                        pickle_posteriors(validPosteriors, self.modelID+'_Posteriors_epoch'+str(epoch)+'_valid')

                testCluster = self.get_clusters_sets(functions[3], batch_reps[2])
                posteriorsTest = [functions[3](i)[1] for i in xrange(batch_reps[2])]
                testPosteriors = [item for sublist in posteriorsTest for item in sublist]
                evaluations[2].create_response(testCluster)
                evaluations[2].print_metrics('Test')
                printer.print_clusters(testCluster, self.data, 'test', settings.elems_to_visualize)
                # getClustersWithFrequenciesTest(testCluster, self.data, settings.elems_to_visualize)

                if debug:
                    pickle_clustering(testCluster, self.modelID+'_epoch'+str(epoch)+'_test')
                    if epoch % 5 == 0 and epoch > 0:
                        pickle_posteriors(testPosteriors, self.modelID+'_Posteriors_epoch'+str(epoch)+'_test')

        endTime = time.clock()
        print 'Optimization complete'
        print 'The code run for {} epochs. Avg epoch duration: {:.1f}s'.format(epoch, (endTime - startTime) / float(epoch))
        print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (endTime - startTime))

    def get_clusters_sets(self, labeling_func, nb_bathces):
        """Build cluster ID => set of examples classified/clustered.\n
        Assigns class labels to the dataset split by maping cluster IDs (int) to sets of example indices (int)\n
        :param labeling_func: label prediction function
        :type labeling_func: callable
        :param nb_bathces: number of batches: batch_size / dataset_size
        :type nb_bathces: int
        :return: the dictionary mapping cluster IDs int => sets with example indices int
        :rtype: dict
        """
        clusters = {}
        for i in xrange(self.relationNum):
            clusters[i] = set()
        # clusters = dict(zip(xrange(self.relationNum), [set()] * self.relationNum))  in toy examples it works. In this system it does not!
        predictions = (item for sublist in (labeling_func(i)[0] for i in xrange(nb_bathces)) for item in sublist)
        for _, pred in enumerate(predictions):
            clusters[pred].add(_)
        return clusters

    @staticmethod
    def get_clusters_size(labeling_func, nb_bathces):
        """Build cluster ID => cluster_size mapping\n
        Computes the population size for each cluster/class/relation using the input train labeling function\n
        :param labeling_func: label prediction function on the 'train' dataset split
        :type labeling_func: callable
        :param nb_bathces: number of bathces; batch_size / dataset_size
        :type nb_bathces: int
        :return: a dictionary mapping classes/cluster2pop IDs (int) to cluster size (int)
        :rtype: dict
        """
        return Counter([item for sublist in map(lambda x: labeling_func(x)[0], xrange(nb_bathces)) for item in sublist])

    def _initialize_optimization_algorithm(self):
        if self.optimization == 'adagrad':
            from learning.Optimizers import AdaGrad
            return AdaGrad(self.modelFunc.params)
        elif self.optimization == 'sgd':
            from learning.Optimizers import SGD
            return SGD()
        else:
            raise Exception("Optimizer '{}' not implemented".format(self.optimization))

    def order_triggers(self, clusters):
        """Build cluster ID => list of sorted triggers\n
        Given the input clustered triggers (dictionary mapping cluster/relations IDs to lists of triggers), builds a dictionary mapping cluster IDs to sorted lists trigger-frequency tuples.
         These lists are sorted in descending order based on frequency values.\n
        :param clusters: clustered triggers
        :type clusters: dict int => list of strings (triggers)
        :return: the sorted clusters
        :rtype: dict int => list
        """
        return dict(zip(xrange(self.relationNum), [sorted(Counter(_).items(), key=operator.itemgetter(1), reverse=True) for _ in clusters.values()]))

    def getClusters(self, labelTrain, trainBatchNum, train_dev):
        clusters = {}
        for i in xrange(self.relationNum):
            clusters[i] = []
        predictionsTrain = [labelTrain(i)[0] for i in xrange(trainBatchNum)]
        predictions = [item for sublist in predictionsTrain for item in sublist]  # returns the flatten() list
        for j in xrange(len(predictions)):
            clusters[predictions[j]].append(self.data.getExampleRelation(j, train_dev))
        return clusters


    def getClusteredFreq(self, clusters):
        clustFreq = {}
        for i in xrange(self.relationNum):
            clustFreq[i] = {}
        j = 0
        for c in clusters:
            for feat in clusters[c]:
                if feat in clustFreq[j]:
                    clustFreq[j][feat] += 1
                else:
                    clustFreq[j][feat] = 1
            clustFreq[j] = sorted(clustFreq[j].iteritems(), key=operator.itemgetter(1), reverse=True)
            j += 1
        return clustFreq

    @staticmethod
    def _mode(data):
        """
        Returns 1, if 'train' split is the only split.\n
        Returns 2, if 'train', 'valid' and 'test' splits are found.\n
        :param data: an instance of DatasetManager
        :type data: learning.OieData.DatasetManager
        :return: {1, 2}
        :rtype: int
        """
        if len(data.split) == 1 and 'train' in data.split:
            return 1
        elif len(
                data.split) == 3 and 'train' in data.split and 'valid' in data.split and 'test' in data.split:
            return 2
        else:
            raise Exception("Either 'train' split or 'train', 'valid' and 'test' splits should be defined")


def saveModel(model, name):
    pklProtocol = 2
    pklFile = open(settings.models_path + name, 'wb')
    pickle.dump(model, pklFile, protocol=pklProtocol)

def loadModel(name):
    pklFile = open(settings.models_path + name, 'rb')
    return pickle.load(pklFile)

def pickleClustering(clustering, clusteringName):
    pklProtocol = 2
    pklFile = open(settings.clusters_path + clusteringName, 'wb')
    pickle.dump(clustering, pklFile, protocol=pklProtocol)


def picklePosteriors(posteriors, posteriorsName):
    pklProtocol = 2
    pklFile = open(settings.clusters_path + posteriorsName, 'wb')
    pickle.dump(posteriors, pklFile, protocol=pklProtocol)

def getClustersWithInfo(clusterSets, data, threshold):
    for c in clusterSets:
        print c,
        if len(clusterSets[c]) < threshold:
            for elem in clusterSets[c]:
                print elem, data.getExampleFeatures(elem),
        else:
            count = 0
            for elem in clusterSets[c]:
                if count > threshold:
                    break
                else:
                    print elem, data.getExampleFeatures(elem),
                    count += 1
        print ''


def getClustersWithRelationLabels(clusterSets, data, evaluation, threshold):
    for c in clusterSets:
        print c,
        if len(clusterSets[c]) < threshold:
            for elem in clusterSets[c]:
                if evaluation.relations[elem][0] != '':
                    print elem, data.get_example_feature(elem), evaluation.relations[elem],
        else:
            count = 0
            for elem in clusterSets[c]:
                if count > threshold:
                    break
                else:
                    if evaluation.relations[elem][0] != '':
                        print elem, data.getExampleFeatures(elem), evaluation.relations[elem],
                        count += 1
        print ''


def save_model(a_model, name):
    """
    Pickles the input oie decoder
    :param a_model: an oie trained decoder
    :type a_model: ReconstructInducer
    :param name: the input decoder name
    :type name: str
    """
    with open(settings.models_path + name, 'wb') as pickled_file:
        pickle.dump(a_model, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(name):
    """
    Unpickles the oie decoder from the input file
    :param name: file containing pickled object, instance of ReconstructInducer
    :type name: str
    :return: the unpickled object
    :rtype: ReconstructInducer
    """
    with open(settings.models_path + name, 'rb') as pickled_file:
        return pickle.load(pickled_file)


def load_data(pickled_dataset, rng, verbose=False):
    """Load from pickled file and index dataset\n
    Unpickles feature extraction functions, Feature Lexicon, dataset batch and goldstanrd batch the file specified in the cmd arguments and wraps them around a DatasetManager\n
    :param pickled_dataset: the file holding the pickled FeatureLexicon object
    :type pickled_dataset: str
    :type rng: numpy.random.RandomState
    :param rng: a random number generator from various distributions
    :param verbose: prints descriptive messages
    :type verbose: bool
    :return: a tuple of the indexed dataset and the goldstandard
    :rtype: learning.OieData.DatasetManager, dict
    """
    if not os.path.exists(pickled_dataset):
        print "Pickled dataset not found"
        sys.exit()
    if verbose:
        print 'Loading data'
    feature_extractions, relation_lexicon, data, gold_standard = unpickle_objects(pickled_dataset)
    indexedDataset = DatasetManager(data, relation_lexicon, rng, verbose=verbose)
    return indexedDataset, gold_standard


def pickle_clustering(clustering, clustering_name):
    with open(settings.clusters_path + clustering_name, 'wb') as pklFile:
        pickle.dump(clustering, pklFile, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_posteriors(posteriors, posteriors_name):
    with open(settings.clusters_path + posteriors_name, 'wb') as pklFile:
        pickle.dump(posteriors, pklFile, protocol=pickle.HIGHEST_PROTOCOL)


def make_shared(matrix_dataset):
    """Converts the inner data structures of a DatasetSplit to the theano shared equivalents\n
    :param matrix_dataset: an instance of class learning.OieData.DatasetSplit
    :return: an instance of class learning.OieData.DatasetSplit
    """
    sharedMatrix = DatasetSplit(
        arguments1=theano.shared(matrix_dataset.args1, borrow=True),
        arguments2=theano.shared(matrix_dataset.args2, borrow=True),
        arg_features=theano.shared(matrix_dataset.xFeats, borrow=True),
    )
    return sharedMatrix


def fix_parsing(bool_flag):
    if bool_flag == 'False' or bool_flag == 'True':
        return eval(bool_flag)
    elif type(bool_flag) == bool:
        return bool_flag
    else:
        raise Exception("Failed to parse '{}' of type '{}'".format(bool_flag, type(bool_flag)))


def get_command_args(program_name):
    parser = argparse.ArgumentParser(prog=program_name, description='Trains a basic Open Information Extraction Model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, help='the pickled dataset file (produced by OiePreprocessor.py)')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=50, help='size of the minibatches')
    parser.add_argument('--embed_size', type=int, default=30, help='embedding space dimensionality')
    parser.add_argument('--relations', type=int, default=3, help='number of semantic relation to induce')
    parser.add_argument('--neg_samples', type=int, default=5, help='number of negative samples to take per entity')
    parser.add_argument('--l1', metavar='lambda_1', type=float, default=0.0, help="value of the 'lambda' L1-norm regularization coefficient")
    parser.add_argument('--l2', metavar='lambda_2', type=float, default=0.0, help="value of the 'lambda' L2-norm regulatization coefficient")
    parser.add_argument('--optimizer', metavar='optimization_algorithm', type=str, default='sgd', help="optimization algorithm one of {'sgd', 'adagrad'}")
    parser.add_argument('--model_name', required=True, type=str, help='Name or ID of the decoder')
    parser.add_argument('--decoder', metavar='decoder', type=str, required=True, help="decoder model; one of {'rescal', 'sp', 'rescal+sp'} for 'RESCAL' bilinear, 'Selectional Preferences', 'RESCAL + SP hybrid'")
    parser.add_argument('--ext_emb', action='store_true', default='False', help='external embeddings')
    parser.add_argument('--ext_reg', action='store_true', default='False', help='regularize decoder model parameters as well')
    parser.add_argument('--freq_eval', action='store_true', default='False', help='use frequent evaluation')
    parser.add_argument('--alpha', metavar='alpha_value', type=float, default=1.0, help='alpha coefficient for scaling the entropy term')
    parser.add_argument('--seed', metavar='seed_number', type=int, default=2, help='random seed')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    arg = parser.parse_args()
    arg.ext_emb = fix_parsing(arg.ext_emb)
    arg.ext_reg = fix_parsing(arg.ext_reg)
    arg.freq_eval = fix_parsing(arg.freq_eval)
    return arg


if __name__ == '__main__':
    print "Relation Learner"

    args = get_command_args(sys.argv[0].split('/')[-1])
    print args
    rand = np.random.RandomState(seed=args.seed)
    indexedData, goldStandard = load_data(args.dataset, rand, verbose=True)

    inducer = ReconstructInducer(indexedData, goldStandard, rand, args.epochs, args.learning_rate, args.batch_size,
                                 args.embed_size, args.relations, args.neg_samples, args.l1, args.l2, args.optimizer, args.model_name, args.decoder,
                                 args.ext_emb, args.ext_reg, args.freq_eval, args.alpha)
    inducer.learn()
    save_model(inducer, inducer.modelName)

    # python learning/OieInduction.py --pickled_dataset sample.pk --model_name discrete-autoencoder --decoder AC --optimization 1 --epochs 10 --batch_size 100 --relations_number 10 --negative_samples_number 5 --l2_regularization 0.1 --alpha 0.1 --seed 2 --embed_size 10 --learning_rate 0.1
