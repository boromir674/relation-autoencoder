import os
import sys
import unittest
import numpy as np
from learning.OieInduction import *

cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset = '/Data/thesis/data/sample-t-v-t.pk'
names = ['ut1', 'Test']

rand = np.random.RandomState(seed=2)


def get_command(dataset_file, model_name):
    return 'python {} {} --ep 2 --l2 0.1 --emb 10 --opt adagrad --dec rescal+sp --alpha 0.1 --model-name {} >/dev/null'.format(cur_dir + '/learning/OieInduction.py', dataset_file, model_name)


def my_try(callback):
    try:
        callback
    except RuntimeError:
        return False
    return True


class TestInductionLearn(unittest.TestCase):

    # def test_mode_1(self):
    #     """Test with given dataset containing only train split"""
    #     failed = False
    #     indexed_data, gold_standard = load_data(datasets[0], rand, verbose=False)
    #     ai_demek = ReconstructInducer(indexed_data, gold_standard, rand, 2, 0.1, 100, 10, 5, 5, 0, 0.1, 'adagrad', names[0], 'rescal+sp', False, True, False, 0.1)
    #     try:
    #         ai_demek.train()
    #     except RuntimeError:
    #         failed = True
    #     if not failed:
    #         ai_demek.initialize()
    #         ai_demek.modelName = names[1]
    #         ai_demek.frequentEval = True
    #         try:
    #             ai_demek.train()
    #         except RuntimeError:
    #             failed = True
    #     self.assertEqual(failed, False)

    def test_mode_2(self):
        """Test with given dataset containing 'train', 'valid', 'test' splits"""
        exit_code = os.system(get_command(dataset, 'model_unit_test')) >> 8
        self.assertEqual(exit_code, 0)

if __name__ == '__main__':
    sys.stdout = open(os.devnull, 'w')
    unittest.main()
    sys.stdout = sys.__stdout__
    close(os.devnull)
