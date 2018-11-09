import tensorflow as tf

import tensorflow.contrib.eager as tfe
from six.moves import xrange
import itertools


def _bundle(lstm_iter):
  """Concatenate a list of Tensors along 1st axis and split result into two.
  Args:
    lstm_iter: A `list` of `N` dense `Tensor`s, each of which has the shape
      (R, 2 * M).
  Returns:
    A `list` of two dense `Tensor`s, each of which has the shape (N * R, M).
  """
  return tf.split(tf.concat(lstm_iter, 0), 2, axis=1)


def _unbundle(state):
  """Concatenate a list of Tensors along 2nd axis and split result.
  This is the inverse of `_bundle`.
  Args:
    state: A `list` of two dense `Tensor`s, each of which has the shape (R, M).
  Returns:
    A `list` of `R` dense `Tensors`, each of which has the shape (1, 2 * M).
  """
  return tf.split(tf.concat(state, 1), state[0].shape[0], axis=0)


# pylint: disable=not-callable
class Reducer(tf.keras.Model):
  """A module that applies reduce operation on left and right vectors."""

  def __init__(self, params, tracker_size=None):        
    size = int(params.embedding_size/2)
    dropout = params.recurrent_dropout
    super(Reducer, self).__init__()
    self.left = tf.keras.layers.Dense(5 * size, activation=None)
    
    self.dropout = tf.keras.layers.Dropout(rate=dropout)
    self.right = tf.keras.layers.Dense(5 * size, activation=None, use_bias=False)
    if tracker_size is not None:
      self.track = tf.keras.layers.Dense(5 * size, activation=None, use_bias=False)
    else:
      self.track = None

  def call(self, left_in, right_in, training, tracking=None):
    """Invoke forward pass of the Reduce module.
    This method feeds a linear combination of `left_in`, `right_in` and
    `tracking` into a Tree LSTM and returns the output of the Tree LSTM.
    Args:
      left_in: A list of length L. Each item is a dense `Tensor` with
        the shape (1, n_dims). n_dims is the size of the embedding vector.
      right_in: A list of the same length as `left_in`. Each item should have
        the same shape as the items of `left_in`.
      tracking: Optional list of the same length as `left_in`. Each item is a
        dense `Tensor` with shape (1, tracker_size * 2). tracker_size is the
        size of the Tracker's state vector.
    Returns:
      Output: A list of length batch_size. Each item has the shape (1, n_dims).
    """
    left, right = _bundle(left_in), _bundle(right_in)
    lstm_in = self.left(left[0]) + self.right(right[0])
    if self.track and tracking:
      lstm_in += self.track(_bundle(tracking)[0])
    return _unbundle(self._tree_lstm(left[1], right[1], lstm_in, training))

  def _tree_lstm(self, c1, c2, lstm_in, training):
    a, i, f1, f2, o = tf.split(lstm_in, 5, axis=1)
    a_ = self.dropout(tf.tanh(a), training = training) 
    c = a_ * tf.sigmoid(i) + tf.sigmoid(f1) * c1 + tf.sigmoid(f2) * c2
    h = tf.sigmoid(o) * tf.tanh(c)
    return h, c


class Tracker(tf.keras.Model):
  """A module that tracks the history of the sentence with an LSTM."""

  def __init__(self, tracker_size, predict):
    """Constructor of Tracker.
    Args:
      tracker_size: Number of dimensions of the underlying `LSTMCell`.
      predict: (`bool`) Whether prediction mode is enabled.
    """
    super(Tracker, self).__init__()
    self._rnn = tf.nn.rnn_cell.LSTMCell(tracker_size)
    self._state_size = tracker_size
    if predict:
      self._transition = tf.keras.layers.Dense(4)
    else:
      self._transition = None

  def reset_state(self):
    self.state = None

  def call(self, bufs, stacks):
    """Invoke the forward pass of the Tracker module.
    This method feeds the concatenation of the top two elements of the stacks
    into an LSTM cell and returns the resultant state of the LSTM cell.
    Args:
      bufs: A `list` of length batch_size. Each item is a `list` of
        max_sequence_len (maximum sequence length of the batch). Each item
        of the nested list is a dense `Tensor` of shape (1, d_proj), where
        d_proj is the size of the word embedding vector or the size of the
        vector space that the word embedding vector is projected to.
      stacks: A `list` of size batch_size. Each item is a `list` of
        variable length corresponding to the current height of the stack.
        Each item of the nested list is a dense `Tensor` of shape (1, d_proj).
    Returns:
      1. A list of length batch_size. Each item is a dense `Tensor` of shape
        (1, d_tracker * 2).
      2.  If under predict mode, result of applying a Dense layer on the
        first state vector of the RNN. Else, `None`.
    """
    buf = _bundle([buf[-1] for buf in bufs])[0]
    stack1 = _bundle([stack[-1] for stack in stacks])[0]
    stack2 = _bundle([stack[-2] for stack in stacks])[0]
    x = tf.concat([buf, stack1, stack2], 1)
    if self.state is None:
      batch_size = int(x.shape[0])
      zeros = tf.zeros((batch_size, self._state_size), dtype=tf.float32)
      self.state = [zeros, zeros]
    _, self.state = self._rnn(x, self.state)
    unbundled = _unbundle(self.state)
    if self._transition:
      return unbundled, self._transition(self.state[0])
    else:
      return unbundled, None


class SPINN(tf.keras.Model):
  """Stack-augmented Parser-Interpreter Neural Network.
  See https://arxiv.org/abs/1603.06021 for more details.
  """

  def __init__(self, params):
    """Constructor of SPINN.
    Args:
      config: A `namedtupled` with the following attributes.
        d_proj - (`int`) number of dimensions of the vector space to project the
          word embeddings to.
        d_tracker - (`int`) number of dimensions of the Tracker's state vector.
        d_hidden - (`int`) number of the dimensions of the hidden state, for the
          Reducer module.
        n_mlp_layers - (`int`) number of multi-layer perceptron layers to use to
          convert the output of the `Feature` module to logits.
        predict - (`bool`) Whether the Tracker will enabled predictions.
    """
    super(SPINN, self).__init__()
    self.params = params
    self.reducer = Reducer(params)    
    self.tracker = None

  def call(self, buffers, transitions, training=False):
    """Invoke the forward pass of the SPINN model.
    Args:
      buffers: Dense `Tensor` of shape
        (max_sequence_len, batch_size, config.d_proj).
      transitions: Dense `Tensor` with integer values that represent the parse
        trees of the sentences. A value of 2 indicates "reduce"; a value of 3
        indicates "shift". Shape: (max_sequence_len * 2 - 3, batch_size).
      training: Whether the invocation is under training mode.
    Returns:
      Output `Tensor` of shape (batch_size, config.d_embed).
    """
    max_sequence_len, batch_size, d_proj = (int(x) for x in buffers.shape)

    # Split the buffers into left and right word items and put the initial
    # items in a stack.
    splitted = tf.split(
        tf.reshape(tf.transpose(buffers, [1, 0, 2]), [-1, d_proj]),
        max_sequence_len * batch_size, axis=0)
    buffers = [splitted[k:k + max_sequence_len]
               for k in xrange(0, len(splitted), max_sequence_len)]
    stacks = [[buf[0], buf[0]] for buf in buffers]


    num_transitions = transitions.shape[0]

    # Iterate through transitions and perform the appropriate stack-pop, reduce
    # and stack-push operations.
    transitions = transitions.numpy()
    for i in xrange(num_transitions):
      trans = transitions[i]
      
      lefts, rights, trackings = [], [], []
      for transition, buf, stack in zip(
          trans, buffers, stacks):
        if int(transition) == 1:  # Shift.
          stack.append(buf.pop())
        elif int(transition) == 2:  # Reduce.
          rights.append(stack.pop())
          lefts.append(stack.pop())
          #trackings.append(tracking)

      if rights:
        reducer_output = self.reducer(lefts, rights, training = training)
        reduced = iter(reducer_output)

        for transition, stack in zip(trans, stacks):
          if int(transition) == 2:  # Reduce.
            stack.append(next(reduced))
    return _bundle([stack.pop() for stack in stacks])[0]