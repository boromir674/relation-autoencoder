import os

# root_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = '/Data/thesis/code/relation-autoencoder/'

models_path = root_dir + '/train_products/'
clusters_path = root_dir + '/train_products/'
posteriors_path = root_dir + '/train_products/'
retreival_metrics = root_dir + '/train_products/'
plots_path = '/Data/thesis/plots/'

# directory in which pickled files are located for unit testing purposes
test_dir = '/Data/thesis/data/'

lda_pairs_path = ''
relations2IdDictionary = ''


external_embeddings_path = ''
debug = True

elems_to_visualize = 5

low = -1.e-3
high = 1.e-3

split_labels = ['train', 'valid', 'test']
