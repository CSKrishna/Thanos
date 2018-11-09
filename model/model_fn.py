"""Define the model."""

import tensorflow as tf
from model.model_parts import *
#import tensorflow.contrib.layers as layers
from model.SPINN import *
from model.evaluation import *
import time
import itertools
import tensorflow.contrib.eager as tfe


class Worddrop_Embedder(tf.keras.Model):
    def __init__(self, params, embeddings_matrix):
        super(Worddrop_Embedder, self).__init__()     
        self.embedding_matrix = tfe.Variable(embeddings_matrix) 
        self.embed_dropout = tf.keras.layers.Dropout(rate=params.word_dropout, noise_shape=[params.vocab_size,1])
        
            
    def call(self, sentences, training):
        word_drops = self.embed_dropout(self.embedding_matrix, training = training)
        return tf.nn.embedding_lookup(word_drops, sentences)
        
        
       
        
        

#generates document level embeddings using SPINN for word level inputs -> sentence embeddings,  GRU RNNs with attention for sentence level embeddings -> document embeddings
class THANOS(tf.keras.Model):
    
    def __init__(self, params, embeddings_matrix):
        """
          Constructor for THANOS.
        """
        super(THANOS, self).__init__()
        self.params = params    
        self.sentence_encoder = SPINN(params)
        self.word_drops = Worddrop_Embedder(params, embeddings_matrix)
        
        self.document_embedder = Document_embedder(params)     
        
        
        self.embed_dropout = tf.keras.layers.Dropout(rate=params.layer_dropout)
        
        self.sentence_output_dropout = tf.keras.layers.Dropout(rate=params.layer_dropout)        
               
        self.attention = Task_specific_attention(params)
        
        self.final_dropout = tf.keras.layers.Dropout(rate=params.layer_dropout)     
              
        self.final = tf.keras.layers.Dense(5, kernel_initializer=tf.random_uniform_initializer(minval=-5e-3,
                                                         maxval=5e-3))
        
   
    def call(self,inputs, training= True):
        sentences = inputs['sentences']
        document_sizes = inputs['document_sizes']
        sentence_lengths = inputs['sentence_lengths']  
        transitions = inputs['transitions']
        is_training = training
        
        
       
        sentence_embeddings = self.word_drops(sentences, training= is_training)    
        sentence_embeddings = self.embed_dropout(sentence_embeddings, training= is_training)
        
        assert(transitions.shape[0] == 2*int(sentence_embeddings.shape[0]) - 3)
        assert(transitions.shape[1] == sentence_embeddings.shape[1])
        
        sentence_level_outputs = self.sentence_encoder(sentence_embeddings, transitions,
                           training = is_training)
        
        #generate document level embeddings
        dim0, dim1 = (int(x) for x in sentence_lengths.shape)        
        sentence_level_outputs  = tf.reshape(sentence_level_outputs,[dim0, dim1, -1])        
       
        sentence_level_outputs = self.sentence_output_dropout(sentence_level_outputs, training = is_training)
           
        
               
        doc_outputs = self.document_embedder(sentence_level_outputs, training = is_training) 

        
        final_outputs = self.attention(doc_outputs, document_sizes)
        
        final_outputs = self.final_dropout(final_outputs, training = is_training)                                                     
    
        # Compute logits from the output of the LSTM
        logits = self.final(final_outputs)
        
        return logits
        
        
class THANOSTrainer(tfe.Checkpointable):        
    
    def __init__(self, THANOS_classifier, params):
        """Constructor of SNLIClassifierTrainer.
        Args:
        snli_classifier: An instance of `SNLIClassifier`.
        lr: Learning rate.
        """
        self._model = THANOS_classifier       
        self._learning_rate = params.learning_rate
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate)
        self.params = params
    
    
    def loss(self, labels, logits):
         return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits))
    
    def train_batch(self, inputs):
        #is_training = (mode == 'train')
        labels = inputs['labels']
        with tf.GradientTape() as tape:
            tape.watch(self._model.variables)
            logits = self._model(inputs, training = True)                           
            loss = self.loss(labels, logits)
            if self.params.l2_reg:
                reg = sum([tf.nn.l2_loss(v) for v in self._model.variables])*self.params.l2_lambda
                loss += reg
           
        gradients = tape.gradient(loss, self._model.variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.variables),
                                    global_step=tf.train.get_global_step())     
        return loss, logits
    
        
    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def model(self):
        return self._model

    @property
    def variables(self):
        return (self._model.variables + [self.learning_rate] +
            self._optimizer.variables())   

            
def _batch_n_correct(logits, label):
  """Calculate number of correct predictions in a batch.
  Args:
    logits: A logits Tensor of shape `(batch_size, num_categories)` and dtype
      `float32`.
    label: A labels Tensor of shape `(batch_size,)` and dtype `int64`
  Returns:
    Number of correct predictions.
  """
  return tf.reduce_sum(
      tf.cast((tf.equal(
          tf.argmax(logits, axis=1), label)), tf.float32)).numpy()    

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



def train_or_infer_spinn(vocab, trans, params, train_dataset, val_dataset, model_dir, embeddings_matrix):
                                 
    use_gpu = tfe.num_gpus() > 0
    device = "gpu:0" if use_gpu else "cpu:0"
    print("Using device: %s" % device)  
                                 
    train_len = params.train_size
                                 
    log_header = (
      "  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss"
      "     Accuracy  Dev/Accuracy")
                                 
    log_template = (
      "{:>6.0f} {:>5.0f} {:>9.0f} {:>5.0f}/{:<5.0f} {:>7.0f}% {:>8.6f} {} "
      "{:12.4f} {}")
                                 
    dev_log_template = (
      "{:>6.0f} {:>5.0f} {:>9.0f} {:>5.0f}/{:<5.0f} {:>7.0f}% {:>8.6f} "
      "{:8.6f} {:12.4f} {:12.4f}")           
                                 
    summary_writer = tf.contrib.summary.create_file_writer(
      model_dir, flush_millis=10000)
                                 
    with tf.device(device), \
       summary_writer.as_default(), \
        tf.contrib.summary.always_record_summaries():
        model = THANOS(params, embeddings_matrix)
        global_step = tf.train.get_or_create_global_step()
        trainer = THANOSTrainer(model, params)
        checkpoint = tf.train.Checkpoint(trainer=trainer, global_step=global_step)
        checkpoint.restore(tf.train.latest_checkpoint(model_dir))
        best_save_path = model_dir+'/'+'best_weights' 
                                 
        start = time.time()
        iterations = 0
        mean_loss = tfe.metrics.Mean()
        accuracy = tfe.metrics.Accuracy()
        logging.info(log_header)
        best_eval_acc = 0.0                        
        for epoch in xrange(params.num_epochs):
            batch_idx = 0
            for labels, document_sizes, sentence_lengths, sentences, transitions in tfe.Iterator(train_dataset):
                #print (sentences)
                if use_gpu:
                    labels, document_sizes, sentence_lengths, sentences, transitions =  labels.gpu(), document_sizes.gpu(), sentence_lengths.gpu(), sentences.gpu(), transitions.gpu()              
          
                inputs = {
                  'labels': labels,
                  'document_sizes': document_sizes,
                  'sentence_lengths': sentence_lengths,
                  'sentences': sentences,
                  'transitions': transitions }
                                 
                iterations += 1
                batch_train_loss, batch_train_logits = trainer.train_batch(inputs)
                batch_size = tf.shape(labels)[0]
                mean_loss(batch_train_loss.numpy(),
                  weights=batch_size.gpu() if use_gpu else batch_size)
                accuracy(tf.argmax(batch_train_logits, axis=1), tf.cast(labels, tf.int64))

                if iterations % params.save_every == 0:
                    checkpoint.save(os.path.join(model_dir, "ckpt"))

                if iterations % params.dev_every == 0:
                    dev_loss, dev_frac_correct = _evaluate_on_dataset(
              val_dataset, trainer, use_gpu)
                    #print(dev_log_template.format(
                          #time.time() - start,
                          #epoch, iterations, 1 + batch_idx, train_len,
                        #100.0 * (1 + batch_idx) / train_len,
                       #mean_loss.result(), dev_loss,
                      #accuracy.result() * 100.0, dev_frac_correct * 100.0))
                    
                    logging.info(dev_log_template.format(
                          time.time() - start,
                          epoch, iterations, 1 + batch_idx, train_len,
                        100.0 * (1 + batch_idx) / train_len,
                       mean_loss.result(), dev_loss,
                      accuracy.result() * 100.0, dev_frac_correct * 100.0))
                    tf.contrib.summary.scalar("dev/loss", dev_loss)
                    tf.contrib.summary.scalar("dev/accuracy", dev_frac_correct)
                    
                    if dev_frac_correct >= best_eval_acc:
                        #best_save_path = model_dir+'/'+'best_weights' 
                        logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                        checkpoint.save(os.path.join(best_save_path, "ckpt_best"))
                        best_eval_acc = dev_frac_correct                        
                                                         
                elif iterations % params.log_every == 0:
                    mean_loss_val = mean_loss.result()
                    accuracy_val = accuracy.result()
                    logging.info(log_template.format(
                          time.time() - start,
                          epoch, iterations, 1 + batch_idx, train_len,
                          100.0 * (1 + batch_idx) / train_len,
                          mean_loss_val, " " * 8, accuracy_val * 100.0, " " * 12))
                    
                    #print(log_template.format(
                          #time.time() - start,
                          #epoch, iterations, 1 + batch_idx, train_len,
                          #100.0 * (1 + batch_idx) / train_len,
                          #mean_loss_val, " " * 8, accuracy_val * 100.0, " " * 12))
                            
                    tf.contrib.summary.scalar("train/loss", mean_loss_val)
                    tf.contrib.summary.scalar("train/accuracy", accuracy_val)
                    
                batch_idx += 1
                # Reset metrics.
                mean_loss = tfe.metrics.Mean()
                accuracy = tfe.metrics.Accuracy()
                
          

    return trainer                            
                                 
                                 
     
                         
                       
                                 
                                 
