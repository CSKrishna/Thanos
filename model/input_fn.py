"""Create the input data pipeline using `tf.data`"""

import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np



def load_dataset_from_text(path_dataset):
    """Create tf.data Instance from python pickle file

    Args:
        path_txt: (string) path to pickle file
   
    Returns:
        dataset: (tf.Dataset) yielding a document, label pair
    """
       
    # Load and pre prepare the dataset    
    data_train = pd.read_pickle(path_dataset)
    
    docs = data_train[["Sent"]].values.tolist()
    docs = [doc[0] for doc in docs]
    
    trees = data_train[["Tree"]].values.tolist()
    trees = [tree[0] for tree in trees]
    
    labels = data_train[["rating"]].values.tolist()    
    labels = [label[0] for label in labels]
    
    def gen():
        for label, doc, tree in zip(labels, docs, trees):
            yield label, doc, tree
            
    ds = tf.data.Dataset.from_generator(gen, (tf.int32, tf.string, tf.string), ([], [None], [None]))  
 
    return ds


def prepare_batch(ds, vocab, trans, params):
    """Input function HAN

    Args:   
        ds: tf.data instance where each element comrrises a document, label tuple
        vocab: pvocab: (tf.lookuptable)
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)

    """
    # Load all the dataset in memory for shuffling is training
  
    
    #extract SHIFT-REDUCE transitions as a string from S-expression
    def transition_parser(st):        
        tokens = st.split()
        string = "" 
        for tok in tokens:
            if tok == ")" :
                string += "REDUCE "
            elif tok == "(":
                pass
            else:
                string += "SHIFT "                
        return string
           
    def _read_py_function(label, sentences, trees):     
        doc_len = len(sentences)
        label = label - 1
        sen_len = [len(str(sentence).split(" ")) for sentence in sentences]  
        trees_ = [tree.decode('utf-8') for tree in trees]        
        transitions = [transition_parser(tree) for tree in trees_] 
        return label, doc_len, sen_len, sentences, transitions
    
    ds = ds.map(lambda label, sentences, trees : tuple(tf.py_func(
     _read_py_function, [label, sentences, trees], [tf.int32, tf.int32, tf.int32, tf.string, tf.string])), num_parallel_calls=4)
    
    #replace tokens with ids
    def transform(doc, default_value = params.pad_word):      
        # Split sentence
        out = tf.string_split(doc) 
        
        # Convert to Dense tensor, filling with default value
        out = tf.sparse_tensor_to_dense(out, default_value=default_value)   
    
        out = vocab.lookup(out)
        out = tf.cast(out, tf.int32)
        return out
    
    #replace SHIFT with 1, REDUCE with 2, all other entries are dummy paddings
    def transform2(doc, default_value = params.pad_word):      
        # Split sentence
        out = tf.string_split(doc) 
        
        # Convert to Dense tensor, filling with default value
        out = tf.sparse_tensor_to_dense(out, default_value=default_value)   
    
        out = trans.lookup(out)
        out = tf.cast(out, tf.int32)
        out += 1
        return out
    
    ds= ds.map(lambda label, doc_len, sen_len, sentences, transitions : (label, doc_len, sen_len, transform(sentences), transform2(transitions)), num_parallel_calls=4)

    # Create batches and pad the sentences of different length
    padded_shapes = (tf.TensorShape([]),
                     tf.TensorShape([]),   # doc of unknown size
                     tf.TensorShape([None]),  # sentence lengths
                     tf.TensorShape([None, None]), # sentence tokens
                     tf.TensorShape([None, None])) # transition tokens
    
    padding_values = (0,0,0,params.id_pad_word,0)                  

    ds = (ds
        .padded_batch(params.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
        .prefetch(4))  # make sure you always have one batch ready to serve     
    
    
    #SPINN Preprocessing step: reverse the entries in sentences and prepend a dummy token
    def pad_reverse(label, doc_len, sen_len, sentences, transitions): 
        # Reverse sequence and pad an extra one.
        sentences_ = np.reshape(sentences, (sentences.shape[0]*sentences.shape[1],sentences.shape[2]))
        sentences_ = np.fliplr(np.array(sentences_, dtype=np.int32))
        sentences_ = np.concatenate([np.ones([sentences_.shape[0] , 1], dtype=np.int32)*params.id_pad_word, sentences_], axis=1)
        sentences_ = sentences_.T
        transitions = np.reshape(transitions, (transitions.shape[0]*transitions.shape[1],-1)).T            
        return label, doc_len, sen_len, sentences_, transitions
  
    ds = ds.map(lambda label, doc_len, sen_len, sentences, transitions : tuple(tf.py_func(
     pad_reverse, [label, doc_len, sen_len, sentences, transitions], [tf.int32, tf.int32, tf.int32, tf.int32, tf.int32])), num_parallel_calls=4)
    
    """

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = ds.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    labels, document_sizes, sentence_lengths, sentences, transitions =  iterator.get_next()
    init_op = iterator.initializer    
    inputs = {
        'labels': labels,
        'document_sizes': document_sizes,
        'sentence_lengths': sentence_lengths,
        'sentences': sentences,
        'transitions': transitions,
        'iterator_init_op': init_op
       }

    return inputs
    """
    return ds

def prepare_dataset(ds, vocab, trans, params):
    #ds = load_dataset_from_text(path_data)
    ds = prepare_batch(ds, vocab, trans, params)
    return ds



