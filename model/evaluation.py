"""Tensorflow utility functions for evaluation"""

import logging
import os

from tqdm import trange
import tensorflow as tf

from model.utils import save_dict_to_json

import tensorflow.contrib.eager as tfe


def _evaluate_on_dataset(val_dataset, trainer, use_gpu):
    mean_loss = tfe.metrics.Mean()
    accuracy = tfe.metrics.Accuracy()
    for labels, document_sizes, sentence_lengths, sentences, transitions in tfe.Iterator(val_dataset):
        if use_gpu:
            labels, document_sizes, sentence_lengths, sentences, transitions =  labels.gpu(), document_sizes.gpu(), sentence_lengths.gpu(), sentences.gpu(), transitions.gpu()
        inputs = {
        'labels': labels,
        'document_sizes': document_sizes,
        'sentence_lengths': sentence_lengths,
        'sentences': sentences,
        'transitions': transitions }
         
         
        logits = trainer.model(inputs, training = False)
        loss_val = trainer.loss(labels, logits)
        batch_size = tf.shape(labels)[0]
        mean_loss(loss_val, weights=batch_size.gpu() if use_gpu else batch_size)
        accuracy(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64))
    return mean_loss.result().numpy(), accuracy.result().numpy()
    

