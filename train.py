"""Train the model"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import tensorflow.contrib.eager as tfe
from model.input_fn import *
import logging
from model.utils import *
from model.model_fn import *


#layers = tf.keras.layers

def main(_):
    
    tf.set_random_seed(420)
    
    #config = FLAGS
    
    json_path = os.path.join(FLAGS.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    
    # Load the parameters from the dataset, that gives the size etc. into params
    json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    params.update(json_path)
    num_oov_buckets = params.num_oov_buckets # number of buckets for unknown words
    
    # Set the logger
    set_logger(os.path.join(FLAGS.model_dir, 'train.log'))

    
    # Get paths for vocabularies and dataset
    path_words = os.path.join(FLAGS.data_dir, 'words.txt')
  
    path_train_sentences = os.path.join(FLAGS.data_dir, 'train.pkl')
    
    path_eval_sentences = os.path.join(FLAGS.data_dir, 'dev.pkl')
    
    
    # Load Vocabularies
    words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=1)
    
    # Get paths for transition codes
    path_trans = os.path.join(FLAGS.data_dir, 'Transitions.txt')
    
    # Load Transition codes
    trans = tf.contrib.lookup.index_table_from_file(path_trans, num_oov_buckets=1)


    # Create the input data pipeline
    logging.info("Creating the datasets...")
    
    train_sentences = load_dataset_from_text(path_train_sentences)

    eval_sentences = load_dataset_from_text(path_eval_sentences)
    
    logging.info("- done.")
    
    
    # Specify other parameters for the dataset and the model
    params.eval_size = params.dev_size
    #params.buffer_size = params.train_size # buffer size for shuffling
    params.id_pad_word = tf.cast(words.lookup(tf.constant(params.pad_word)), tf.int32)
    
    # Create the tf.data.dataset structures over the two datasets
    train_ds = prepare_dataset(train_sentences, words, trans, params)
    val_ds = prepare_dataset(eval_sentences, words, trans, params)   
    
     #load pretrained Glove vectors
    filtered_glove_path = os.path.join(FLAGS.data_dir, 'filtered_glove.txt')
    weight_matrix, word_idx = load_embeddings(filtered_glove_path)
    embeddings_matrix = create_embeddings_matrix(path_words, weight_matrix, word_idx)
    
  
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    
    train_or_infer_spinn(words, trans, params, train_ds, val_ds, FLAGS.model_dir, embeddings_matrix)


if __name__ == "__main__":   
    
    parser = argparse.ArgumentParser(description=
      "TensorFlow eager implementation of the SPINN SNLI classifier.")
    parser.add_argument('--model_dir', default='experiments/Yelp_11_04',
                    help="Directory containing params.json")
    parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
    parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")

    FLAGS, unparsed = parser.parse_known_args()
    tfe.run(main=main, argv=[sys.argv[0]] + unparsed)



