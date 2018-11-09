import tensorflow as tf
import tensorflow.contrib.layers as layers

class Document_embedder(tf.keras.Model):
    def __init__(self, params, bidirectional = False):
        super(Document_embedder, self).__init__()
        self.embed = tf.keras.layers.GRU(params.GRU_num_units, return_sequences=True, recurrent_dropout = params.recurrent_dropout)
        self.bidirectional = bidirectional
        if bidirectional:
            self.biembed = tf.keras.layers.Bidirectional(self.embed)
        
    def call(self, sentece_embeds, training):
        if self.bidirectional:
            outputs = self.biembed(sentece_embeds, training = training)
        else:
            outputs = self.embed(sentece_embeds, training = training)
        return outputs
    


    

        
class Task_specific_attention(tf.keras.Model):
    def __init__(self, params):
        super(Task_specific_attention, self).__init__()
        
        self.attention = tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform', use_bias=False)
    
        self.projection = tf.keras.layers.Dense(params.GRU_num_units, activation=tf.tanh, kernel_initializer='glorot_uniform')
        
    def call(self, doc_outputs, sequence_lengths):
        input_projection = self.projection(doc_outputs)        
     
        vector_attn = tf.squeeze(self.attention(input_projection))
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)                                        
        attention_weights = tf.nn.softmax(vector_attn, axis=1)
        attention_weights = attention_weights*mask
        norms = tf.reduce_sum(attention_weights, axis = 1, keepdims = True) + 1e-6     
        attention_weights = attention_weights / norms
        attention_weights = tf.expand_dims(attention_weights, axis = 2)        
        
        weighted_projection = doc_outputs*attention_weights
        outputs = tf.reduce_sum(weighted_projection, axis=1)
        
        return outputs

        
    



