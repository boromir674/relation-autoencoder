import math
import theano
import numpy as np
import theano.tensor as T
from Decoder import Decoder


class SelectionalPreferences(Decoder):

    def __init__(self, rng, neg_samples_num, batch_size, embedSize, relationNum, argVocSize, ex_emb):
        super(SelectionalPreferences, self).__init__(rng, neg_samples_num, batch_size, embedSize, relationNum, argVocSize, ex_emb)
        # Selectional Preferences of arguments/entities 1 and 2
        Ca1NP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(self.r, self.m)), dtype=theano.config.floatX)  # [ c_{11}, c_{12}, c_{13}, ..., c_{1m} ]
        Ca2NP = np.asarray(rng.normal(0, math.sqrt(0.1), size=(self.r, self.m)), dtype=theano.config.floatX)  # [ c_{21}, c_{22}, c_{23}, ..., c_{2m} ]

        self.C1 = theano.shared(value=Ca1NP, name='C1')  # (r,m)
        self.C2 = theano.shared(value=Ca2NP, name='C2')  # (r,m)
        self.A = theano.shared(value=self.A_np, name='A')  # (n,r)
        self.Ab = theano.shared(value=np.zeros(self.n,  dtype=theano.config.floatX), name='Ab', borrow=True)  # (n,)

    def get_parameters(self):
        return [self.A, self.C1, self.C2, self.Ab]

    def get_l1_regularization_term_computation(self):
        return T.sum(abs(self.C1)) + T.sum(abs(self.C2))  # (1,)

    def get_l2_regularization_term_computation(self):
        return T.sum(T.sqr(self.C1)) + T.sum(T.sqr(self.C2))  # (1,)

    def get_scores(self, args1, args2, relation_probs, neg1, neg2, entropy):
        weightedC1 = T.dot(relation_probs, self.C1.dimshuffle(1, 0))  # (l,m) x (m,r) = (l,r)
        weightedC2 = T.dot(relation_probs, self.C2.dimshuffle(1, 0))  # (l,m) x (m,r) = (l,r)

        left_factorization = T.batched_dot(weightedC1, self.A[args1.flatten()])  # (l,r) x (l,r) = (l,)
        right_factorization = T.batched_dot(weightedC2, self.A[args1.flatten()])  # (l,r) x (l,r) = (l,)
        one = left_factorization + right_factorization  # (l,)

        u = T.concatenate([one + self.Ab[args1], one + self.Ab[args2]])  # (2l,)
        allScores = T.concatenate([T.log(T.nnet.sigmoid(u)), entropy, entropy])  # (4l,)

        negembed1 = self.A[neg1.flatten()].reshape((self.s, self.l, self.r))  # (s,l,r)
        negembed2 = self.A[neg2.flatten()].reshape((self.s, self.l, self.r))  # (s,l,r)
        neg_left_factorization = T.batched_tensordot(weightedC1, negembed1.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r) x (l,r,s) = (l,s)
        neg_right_factorization = T.batched_tensordot(weightedC2, negembed2.dimshuffle(1, 2, 0), axes=[[1], [1]])  # (l,r) x (l,r,n) = (l,s)

        negOne = neg_left_factorization.dimshuffle(1, 0) + right_factorization  # (s,l) + (l,) = (s,l)
        negTwo = neg_right_factorization.dimshuffle(1, 0) + left_factorization  # (s,l) + (l,) = (s,l)
        g = T.concatenate([negOne + self.Ab[neg1], negTwo + self.Ab[neg2]])  # (2s,l)
        logScores = T.log(T.nnet.sigmoid(-g))  # (2s,l)
        allScores = T.concatenate([allScores, logScores.flatten()])  # (4l+2ls,)
        return allScores
