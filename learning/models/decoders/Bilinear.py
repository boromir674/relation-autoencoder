import theano
from Decoder import Decoder
import numpy as np
from math import sqrt
import theano.tensor as T

__author__ = 'enfry'


class Bilinear(Decoder):
    """An inmplementation of the RESCAL _factorization decoder"""

    def __init__(self, rng, neg_samples_num, batch_size, r, m, n, A_np):
        super(Bilinear, self).__init__(rng, neg_samples_num, batch_size, r, m, n, A_np)
        RNP = np.asarray(rng.normal(0, sqrt(0.1), size=(r, r, m)), dtype=theano.config.floatX)
        self.R = theano.shared(value=RNP, name='R')  # (r,r,m)
        self.A = theano.shared(value=A_np, name='A')  # (n,r)
        self.Ab = theano.shared(value=np.zeros(self.n,  dtype=theano.config.floatX), name='Ab', borrow=True)  # (n,)

    def get_parameters(self):
        return [self.R, self.A, self.Ab]

    def get_l1_regularization_term_computation(self):
        return T.sum(abs(self.R))  # (1,)

    def get_l2_regularization_term_computation(self):
        return T.sum(T.sqr(self.R))  # (1,)

    def get_scores(self, args1, args2, relation_probs, neg1, neg2, entropy):

        argembed1 = self.A[args1]  # (l,r)
        argembed2 = self.A[args2]  # (l,r)

        weighted_R = T.tensordot(relation_probs, self.R, axes=[[1], [2]])  # (l,m) x (r,r,m) = (l,r,r)
        one = self._factorization(ent1_embeddings=argembed1, ent2_embeddings=argembed2, weighted_r=weighted_R)  # (l,)

        u = T.concatenate([one + self.Ab[args1], one + self.Ab[args2]])  # (2l,)

        logScoresP = T.log(T.nnet.sigmoid(u))  # (2l,)
        allScores = T.concatenate([logScoresP, entropy, entropy])  # (4l,)

        negembed1 = self.A[neg1.flatten()].reshape((self.s, self.l, self.r))  # (s,l,r)
        negembed2 = self.A[neg2.flatten()].reshape((self.s, self.l, self.r))  # (s,l,r)
        negOne = self._neg_factorization1(neg_emb1=negembed1, args_emb2=argembed2, weighted_r=weighted_R)  # (l,s)
        negTwo = self._neg_factorization2(argsEmbA=argembed1, negEmbB=negembed2, wC=weighted_R)  # (l,s)

        g = T.concatenate([negOne + self.Ab[neg1].dimshuffle(1, 0), negTwo + self.Ab[neg2].dimshuffle(1, 0)])  # (2l,s)
        logScores = T.log(T.nnet.sigmoid(-g))  # (2l,s)
        allScores = T.concatenate([allScores, logScores.flatten()])  # (4l+2ls,)
        return allScores

    def _factorization(self, ent1_embeddings, ent2_embeddings, weighted_r):
        """
        :param ent1_embeddings: (l,r)
        :param ent2_embeddings: (l,r)
        :param weighted_r: (l,r,r)
        :return: array (l,)
        """
        Afirst = T.batched_tensordot(weighted_r, ent1_embeddings, axes=[[1], [1]])  # (l,r,r) x (l,r) = (l,r)
        return T.batched_dot(Afirst, ent2_embeddings)  # (l,r) x (l,r) = (l,)

    def _neg_factorization1(self, neg_emb1, args_emb2, weighted_r):
        """
        :param neg_emb1: (s,l,r)
        :param args_emb2: (l,r)
        :param weighted_r: (l,r,r)
        :return: (l,)
        """
        Afirst = T.batched_tensordot(weighted_r, neg_emb1.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r,r) x (l,r,s) = (l,r,s)
        return T.batched_tensordot(Afirst, args_emb2, axes=[[1], [1]])  # (l,r,s) x (l,r) = (l,s)

    def _neg_factorization2(self, argsEmbA, negEmbB, wC):
        """
        :param argsEmbA: (l,r)
        :param negEmbB: (s,l,r)
        :param wC: (l,r,r)
        :return:
        """
        Afirst = T.batched_tensordot(wC, argsEmbA, axes=[[1], [1]])  # (l,r,r) x (l,r) = (l,r)
        return T.batched_tensordot(Afirst, negEmbB.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r) x (l,r,s) = (l,s)
