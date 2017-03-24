import math
import theano
import numpy as np
import theano.tensor as T
from Decoder import Decoder


class BilinearPlusSP(Decoder):

    def __init__(self, rng, neg_samples_num, batch_size, embedSize, relationNum, argVocSize, ex_emb):
        super(BilinearPlusSP, self).__init__(rng, neg_samples_num, batch_size, embedSize, relationNum, argVocSize, ex_emb)
        # for every relation/class model entities relation_labels between each other; m number of r x r matrices;
        CNP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(self.r, self.r, self.m)), dtype=theano.config.floatX)
        # Selectional Preferences
        Ca1NP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(self.r, self.m)), dtype=theano.config.floatX)
        Ca2NP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(self.r, self.m)), dtype=theano.config.floatX)

        self.C = theano.shared(value=CNP, name='C')  # (r,r,m)
        self.A = theano.shared(value=self.A_np, name='A')  # (n,r)
        self.Ab = theano.shared(value=np.zeros(self.n, dtype=theano.config.floatX), name='Ab', borrow=True)  # (n,)
        self.C1 = theano.shared(value=Ca1NP, name='C1')  # (r,m)
        self.C2 = theano.shared(value=Ca2NP, name='C2')  # (r,m)

    def get_l1_regularization_term_computation(self):
        return T.sum(abs(self.C1)) + T.sum(abs(self.C2)) + T.sum(abs(self.C))  # (1,) + (1,) + (1,) = (1,)

    def get_l2_regularization_term_computation(self):
        return T.sum(T.sqr(self.C1)) + T.sum(T.sqr(self.C2)) + T.sum(T.sqr(self.C))  # (1,)

    def get_parameters(self):
        return [self.C, self.A, self.Ab, self.C1, self.C2]

    def get_scores(self, args1, args2, relation_probs, neg1, neg2, entropy):
        weightedC1 = T.dot(relation_probs, self.C1.dimshuffle(1, 0))  # (l,m) x (m,r) = (l,r); C1.dimshuffle(1,0): (m,r)
        weightedC2 = T.dot(relation_probs, self.C2.dimshuffle(1, 0))  # (l,m) x (m,r) = (l,r); C1.dimshuffle(1,0): (m,r)
        weightedC = T.tensordot(relation_probs, self.C, axes=[[1], [2]])  # (l,m) x (r,r,m) = (l,r,r)

        argembed1 = self.A[args1]  # (l,r)
        argembed2 = self.A[args2]  # (l,r)

        one = self._factorization(args_emb_a=argembed1, args_emb_b=argembed2, wC=weightedC, wC1=weightedC1, wC2=weightedC2)  # (l,)

        u = T.concatenate([one + self.Ab[args1], one + self.Ab[args2]])  # (2l,)
        logScoresP = T.log(T.nnet.sigmoid(u))  # (2l,)

        allScores = T.concatenate([logScoresP, entropy, entropy])  # (4l,)

        negembed1 = self.A[neg1.flatten()].reshape((self.s, self.l, self.r))  # (s,l,r)
        negembed2 = self.A[neg2.flatten()].reshape((self.s, self.l, self.r))  # (s,l,r)
        negOne = self._neg_left_factorization(negEmbA=negembed1, argsEmbB=argembed2, wC=weightedC, wC1=weightedC1, wC2=weightedC2)  # (l,s)
        negTwo = self._neg_right_factorization(argsEmbA=argembed1, negEmbB=negembed2, wC=weightedC, wC1=weightedC1, wC2=weightedC2)  # (l,s)

        g = T.concatenate([negOne + self.Ab[neg1].dimshuffle(1, 0), negTwo + self.Ab[neg2].dimshuffle(1, 0)])  # (2l,s)
        logScores = T.log(T.nnet.sigmoid(-g))  # (2l,s)
        allScores = T.concatenate([allScores, logScores.flatten()])  # (4l+2ls,)
        return allScores

    def _factorization(self, args_emb_a, args_emb_b, wC, wC1, wC2):
        """
        :param args_emb_a: (l,r)
        :param args_emb_b: (l,r)
        :param wC: (l,r,r)
        :param wC1: (l,r)
        :param wC2: (l,r)
        :return: (l,)
        """
        Afirst = T.batched_tensordot(wC, args_emb_a, axes=[[1], [1]])  # (l,r,r) x (l,r) = (l,r)
        Asecond = T.batched_dot(Afirst, args_emb_b)  # (l,r) x (l,r) = (l,)
        spFirst = T.batched_dot(wC1, args_emb_a)  # (l,r) x (l,r) = (l,)
        spSecond = T.batched_dot(wC2, args_emb_b)  # (l,r) x (l,r) = (l,)
        return Asecond + spFirst + spSecond  # (l,) + (l,) + (l,) = (l,)

    def _neg_left_factorization(self, negEmbA, argsEmbB, wC, wC1, wC2):
        """
        :param negEmbA: (s,l,r)
        :param argsEmbB: (l,r)
        :param wC: (l,r,r)
        :param wC1: (l,r)
        :param wC2: (l,r)
        :return: (l,s)
        """
        Afirst = T.batched_tensordot(wC, negEmbA.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r,r) x (l,r,s) = (l,r,s)
        Asecond = T.batched_tensordot(Afirst, argsEmbB, axes=[[1], [1]])  # (l,r,s) x (l,r) = (l,s)
        spAfirst = T.batched_tensordot(wC1, negEmbA.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r) x (l,r,s) = (l,s)
        spSecond = T.batched_dot(wC2, argsEmbB)  # (l,r) x (l,r) = (l,)
        return Asecond + spAfirst + spSecond.reshape((self.l, 1))  # (l,s) + (l,s) + (l,1) = (l,s)

    def _neg_right_factorization(self, argsEmbA, negEmbB, wC, wC1, wC2):
        """
        :param argsEmbA: (l,r)
        :param negEmbB: (s,l,r)
        :param wC: (l,r,r)
        :param wC1: (l,r)
        :param wC2: (l,r)
        :return: (l,s)
        """
        Afirst = T.batched_tensordot(wC, argsEmbA, axes=[[1], [1]])  # (l,r,r) x (l,r) = (l,r)
        Asecond = T.batched_tensordot(Afirst, negEmbB.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r) x (l,r,s) = (l,s)
        spFirst = T.batched_dot(wC1, argsEmbA)  # (l,r) x (l,r) = (l,)
        spAsecond = T.batched_tensordot(wC2, negEmbB.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r) x (l,r,s) = (l,s)
        return Asecond + spAsecond + spFirst.reshape((self.l, 1))  # (l,s) + (l,s) + (l,1) = (l,s)
