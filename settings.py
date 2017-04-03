import os

root_dir = os.path.dirname(os.path.realpath(__file__))

models_path = ''
clusters_path = root_dir + '/train_products/'
posteriors_path = root_dir + '/train_products/'
retreival_metrics = root_dir + '/train_products/'
lda_pairs_path = ''
relations2IdDictionary = ''

external_embeddings_path = ''
debug = True

elems_to_visualize = 5

low = -1.e-3
high = 1.e-3
