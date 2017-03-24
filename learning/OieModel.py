import settings
from numpy import zeros
from numpy import asarray
import theano.tensor as T
from theano import config as thc
from models.decoders.Decoder import construct_decoder
from models.encoders.RelationClassifier import IndependentRelationClassifiers

__author__ = 'diego'


class OieModelFunctions(object):
    """
    A class resposible for 'firing' the encoder and the selected decoder.
    """
    def __init__(self, rng, embed_size, nb_relations, neg_samples_num, batch_size, model, data, extended_regularizer, alpha, external_embeddings=False):
        """
        :param rng: random number generator
        :type rng: numpy.random.RandomState
        :param embed_size: user defined size of the entity embeddings; dimensionality of axis 2 of A matrix (r in notes)
        :type embed_size: int
        :param nb_relations: the number of clusters/semantic relations to induce
        :type nb_relations: int
        :param neg_samples_num: the number of samples to take per e1, e2 entities for the 'negative sampling approximation'
        :type neg_samples_num: int
        :param batch_size: the number of datapoints in each batch
        :type batch_size: int
        :param model: decoder model; one of 'rescal', 'sp', 'rescal+sp' for 'RESCAL' bilinear, 'Selectional Preferences', 'RESCAL+SP' hybrid
        :type model: str
        :param data: the encoded dataset
        :type data: learning.OieData.DatasetManager
        :param extended_regularizer: if true adds a regularizing term as a penalty for the decode's weights, in addition to the term for the encoder's weights. If false, uses only the penalty term for the encoder's weights
        :type extended_regularizer: bool
        :param alpha: real number in [0,1]. Aims to scale the entropy term to the expectation of the approximated softmax
        :type alpha: float
        :param external_embeddings: if true initializes entity embeddings with word2vec vectors
        :type external_embeddings: bool
        """
        self.rng = rng
        self.r = embed_size
        self.s = neg_samples_num
        self.l = batch_size
        self.m = nb_relations
        self.n = data.get_arg_voc_size()
        self.model = model
        self.external_emb = external_embeddings
        self.extended_reg = extended_regularizer
        self.alpha = alpha
        self.relationClassifiers = IndependentRelationClassifiers(rng, data.get_dimensionality(), nb_relations)
        self.params = self.relationClassifiers.params

        # W  : (d, m)
        # L1-norm regularizer; (1,)
        self.L1 = T.sum(abs(self.relationClassifiers.W))  # sums the absolute values of flatten tensor
        # L2-norm regularizer; (1,)
        self.L2 = T.sum(T.sqr(self.relationClassifiers.W))  # sums the squared values of flatten tensor

        embds = self._initialize_entity_embeddings(data, self.external_emb)  # (n,r)
        self.decoder = construct_decoder(model, rng, self.s, self.l, embed_size, nb_relations, data.get_arg_voc_size(), embds)
        if self.extended_reg:
            self.L1 += self.decoder.get_l1_regularization_term_computation()
            self.L2 += self.decoder.get_l2_regularization_term_computation()
        self.params.extend(self.decoder.get_parameters())

    def build_train_err_computation(self, x_feats, args1, args2, neg1, neg2):
        """Build the error computation for training.\n
        Given the inut symbolic TensorVariables, constructs the expression for computing the training error to be minimized. This is equivalent to the negation of the reconstruction objective function\n
        :param x_feats: a binary feature triggering matrix of shape (l,d) :: [batch_size, feature_space_dimensionality]
        :param args1: a vector of shape (l,) containing the IDs of the e1 entities
        :type args1: tensor
        :param args2: a vector of shape (l,) containing the IDs of the e2 entities
        :type args2: tensor
        :param neg1: a matrix of shape (s, l): s number of sampled pointers per e1 entity
        :type neg1: tensor
        :param neg2: a matrix of shape (s, l): s number of sampled pointers per e2 entity
        :type neg2: tensor
        :return: the error computation
        :rtype: scalar
        """
        relation_probs = self.relationClassifiers.comp_relation_probs(x_feats=x_feats)  # (l,m) classes probabilities
        entropy = self.alpha * -T.sum(T.log(relation_probs) * relation_probs, axis=1)  # (l,)
        all_scores = self.decoder.get_scores(args1, args2, relation_probs, neg1, neg2, entropy)  # (a*l,)
        # minimize the negative objective function
        res_error = -T.mean(all_scores)  # scalar
        print '  built full computation graph'
        return res_error

    def build_label_computation(self, x_feats):
        """Build the labeling and class probability computation.\n
        Uses the encoder's computing capabilities for labeling and posterior calculating: a wrapper around IndependentRelationClassifiers.comp_probs_and_labels method.\n
        :param x_feats: (l, d) shaped binary (sparse) matrix of feature triggering
        :type x_feats: theano.sparse / symbolic matrix
        :return: a tuple of (l,) and (l, m) shaped tensors of labels and class probabilities, respectively
        """
        return self.relationClassifiers.comp_probs_and_labels(x_feats)

    def _initialize_entity_embeddings(self, data, word2vecflag):
        """Checks the external_embeddings flag and returns a np.array accordingly"""
        A_np = asarray(self.rng.uniform(-0.01, 0.01, size=(data.get_arg_voc_size(), self.r)), dtype=thc.floatX)  # np.asarray
        if word2vecflag == 'True':
            word2vecflag = True
        elif word2vecflag == 'False':
            word2vecflag = False
        if word2vecflag:
            import gensim
            external_embeddings = gensim.models.Word2Vec.load(settings.external_embeddings_path)
            for entity_id in xrange(data.get_arg_voc_size()):
                entity = data.id2Arg[entity_id].lower().split(' ')
                new = zeros(self.r, dtype=thc.floatX)  # np.zeros
                size = 0
                for token in entity:
                    if token in external_embeddings:
                        new += external_embeddings[token]
                        size += 1
                if size > 0:
                    A_np[entity_id] = new / size
        return A_np
