import abc


class Decoder(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, model_type, rng, neg_samples_num, batch_size, embeddings_size, relation_num, entity_vocab_size, init_embds=None):
        """
        :param model_type: the implemented model used in the decodeing unit
        :type model_type: str {'rescal', 'sp', 'rescal+sp'}
        :param rng: a random number generator
        :type rng: numpy.random.RandomState
        :param neg_samples_num: number of samples to take per e1, e2 entity for the 'negative sampling approximation'
        :type neg_samples_num: int
        :param batch_size: the number of datapoints to process in a batch
        :type batch_size: int
        :param embeddings_size: dimensionality of the entities embedding space
        :type embeddings_size: int
        :param relation_num: number of classes/semantic relation to induce
        :type relation_num: int
        :param entity_vocab_size: number of (unique) entity instances found in the dataset
        :type entity_vocab_size: int
        :param init_embds: initial array of entity embeddings, typical zeros unless word2vec vectors loaded
        :type init_embds: numpy.ndarray
        """
        self.type = model_type
        self.rng = rng
        self.s = neg_samples_num
        self.l = batch_size
        self.r = embeddings_size
        self.m = relation_num
        self.n = entity_vocab_size
        self.A_np = init_embds

    @staticmethod
    def creat_decoder_with_same_specs(a_decoder):
        """Constructs a Decoder object initialized according to the input Decoder's specs. Initial embeddings are set to None\n
        :param a_decoder: an instance object
        :type a_decoder: Decoder
        :return: a decoder_type initialized with the same specs as the input decoder_type
        :rtype: Decoder
        """
        return construct_decoder(a_decoder.type, a_decoder.rng, a_decoder.s, a_decoder.l, a_decoder.r, a_decoder.m, a_decoder.n, init_embds=None)

    def set_embeddings(self, initial_embeddings):
        self.A_np = initial_embeddings

    @abc.abstractmethod
    def get_scores(self, args1, args2, relation_probs, neg1, neg2, entropy):
        """Scores the _factorization performance\n
        :param args1: e1 entity IDs: pointers to e1 entities
        :type args1: array (l,)
        :param args2: e2 entity IDs: pointers to e2 entities
        :type args2: array (l,)
        :param relation_probs: class probability distribution array of shape (batch_size, number_of_classes)
        :type relation_probs: array (l,m)
        :param neg1: sampled indices for each e1 entity acting as IDs
        :type neg1: array (s,l)
        :param neg2: sampled indices for each e2 entity acting as IDs
        :type neg2: array (s, l)
        :param entropy: the entropy values computed for all datapoint
        :type entropy: array (l,)
        :return: the objective function score
        :rtype: array
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_l1_regularization_term_computation(self):
        """Returns the L1-norm regularization term\n
        Should return the symbolic computation of the L1-norm ergularization term\n
        :rtype: theano.tensor.scalar
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_l2_regularization_term_computation(self):
        """Returns the L2 regularization term\n
        Should return the symbolic computation of the L2-norm regularization term\n
        :rtype: theano.tensor.scalar
        """
        raise NotImplementedError

    @abc.abstractproperty
    def get_parameters(self):
        """Returns the model parametes\n
        Returns the model parameters as a list of weight tensors
        :return: of tensors
        :rtype: list
        """
        raise NotImplementedError


def construct_decoder(model_type, rng, neg_samples_num, batch_size, embeddings_size, relation_num, entity_vocab_size, init_embds=None):
    if model_type == 'rescal':
        from Bilinear import Bilinear
        return Bilinear(rng, neg_samples_num, batch_size, embeddings_size, relation_num, entity_vocab_size, init_embds)
    elif model_type == 'rescal+sp':
        from BilinearPlusSP import BilinearPlusSP
        return BilinearPlusSP(rng, neg_samples_num, batch_size, embeddings_size, relation_num, entity_vocab_size, init_embds)
    elif model_type == 'sp':
        from SelectionalPreferences import SelectionalPreferences
        return SelectionalPreferences(rng, neg_samples_num, batch_size, embeddings_size, relation_num, entity_vocab_size, init_embds)
