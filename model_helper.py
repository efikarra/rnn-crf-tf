import collections
import tensorflow as tf
import time
import iterator_utils
import vocab_utils
import numpy as np


class TrainModel(collections.namedtuple("TrainModel",("graph", "model", "iterator"))):
    pass


class EvalModel(collections.namedtuple("EvalModel",("graph", "model", "input_file_placeholder", "output_file_placeholder", "iterator"))):
    pass


class InferModel(collections.namedtuple("EvalModel",("graph", "model", "input_file_placeholder", "iterator"))):
    pass


def create_train_model(model_creator, hparams, input_path, target_path, mode):
    """Create train graph, model, and iterator."""
    graph = tf.Graph()
    with graph.as_default() , tf.container("train"):
        # create a table to map vocabulary words to vocab ids.
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        input_dataset = tf.data.TextLineDataset(input_path)
        output_dataset = tf.data.TextLineDataset(target_path)
        # create iterator over the train batches.
        iterator = iterator_utils.get_iterator(input_dataset, output_dataset, input_vocab_table,
                                               batch_size=hparams.batch_size, random_seed=hparams.random_seed,
                                               pad=hparams.pad, input_max_len=hparams.input_max_len)

        # create the actual rnn model.
        model = model_creator(hparams, mode, iterator, input_vocab_table=input_vocab_table)
        return TrainModel(graph, model, iterator)


def create_eval_model(model_creator, hparams, mode):
    """Create eval graph, model, src/tgt file holders, and iterator."""
    graph = tf.Graph()
    with graph.as_default(), tf.container("eval"):
        # create a table to map words to vocab ids.
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        # define a placeholder for the input dataset.
        # we will dynamically initialize this placeholder with a file name during validation.
        # The reason for this is that during validation, we may want to evaluate our trained model on different datasets.
        input_file_placeholder= tf.placeholder(shape=(),dtype=tf.string)
        input_dataset = tf.data.TextLineDataset(input_file_placeholder)
        output_file_placeholder= tf.placeholder(shape=(), dtype=tf.string)
        output_dataset = tf.data.TextLineDataset(output_file_placeholder)

        iterator = iterator_utils.get_iterator(input_dataset, output_dataset, input_vocab_table,
                                               batch_size=hparams.eval_batch_size, random_seed=hparams.random_seed,
                                               pad=hparams.pad, input_max_len=hparams.input_max_len)
        model = model_creator(hparams, mode, iterator, input_vocab_table=input_vocab_table)
        return EvalModel(graph, model, input_file_placeholder, output_file_placeholder, iterator)


def create_infer_model(model_creator, hparams, mode):
    """Create inference model."""
    graph = tf.Graph()
    with graph.as_default(), tf.container("predict"):
        # create a table to map words to vocab ids.
        input_vocab_table = vocab_utils.create_vocab_table(hparams.vocab_path)
        input_file_placeholder= tf.placeholder(shape=(),dtype=tf.string)
        input_dataset = tf.data.TextLineDataset(input_file_placeholder)

        iterator = iterator_utils.get_iterator_infer(input_dataset, input_vocab_table,
                                                     batch_size=hparams.predict_batch_size, pad=hparams.pad,
                                                     input_max_len=hparams.input_max_len)
        model = model_creator(hparams, mode, iterator, input_vocab_table=input_vocab_table)
        return InferModel(graph, model, input_file_placeholder, iterator)


def create_embeddings(vocab_size,emb_size,emb_trainable,emb_pretrain,dtype=tf.float32):
    """Create embedding layer."""
    with tf.variable_scope("embeddings", dtype=dtype):
        emb_placeholder=None
        emb_init=None
        embedding = tf.get_variable("embedding", [vocab_size, emb_size], dtype, trainable=emb_trainable)
        if emb_pretrain:
            emb_placeholder = tf.placeholder(tf.float32, [vocab_size, emb_size])
            emb_init = embedding.assign(emb_placeholder)
        return embedding, emb_init, emb_placeholder


def _single_cell(unit_type, num_units, forget_bias, in_to_hidden_dropout):

    # Cell Type
    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units,forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units,forget_bias=forget_bias,layer_norm=True)
    elif unit_type == "rnn":
      single_cell = tf.contrib.rnn.BasicRNNCell(
          num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)
    # Dropout (= 1 - keep_prob)
    if in_to_hidden_dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - in_to_hidden_dropout))
    return single_cell


def _cell_list(unit_type, num_units, num_layers, forget_bias, in_to_hidden_dropout, mode):
  """Create a list of RNN cells if the model has num_layers > 1."""
  cell_list = []
  for i in range(num_layers):
    in_to_hidden_dropout = in_to_hidden_dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
    single_cell = _single_cell(
        unit_type=unit_type,
        num_units=num_units,
        forget_bias=forget_bias,
        in_to_hidden_dropout=in_to_hidden_dropout)
    cell_list.append(single_cell)
  return cell_list


def create_rnn_cell(unit_type, num_units, num_layers, forget_bias, in_to_hidden_dropout, mode):
    """Create multi-layer RNN cell.
      Args:
        unit_type: string representing the unit type, i.e. "lstm".
        num_units: the size of each unit.
        num_layers: number of layers.
        forget_bias: the initial forget bias of the RNNCell(s).
        in_to_hidden_dropout: floating point value between 0.0 and 1.0: dropout from input to hidden layer.
          this is ignored if `mode != TRAIN`.
        mode: either tf.contrib.learn.TRAIN/EVAL/INFER
      Returns:
        An `RNNCell` instance.
      """
    cell_list = _cell_list(unit_type=unit_type,
                     num_units=num_units,
                     num_layers=num_layers,
                     forget_bias=forget_bias,
                     in_to_hidden_dropout=in_to_hidden_dropout,
                     mode=mode)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
    # if the global_norm, i.e., the sum of the norms of all gradients, exceeds max_gradient_norm
    # then clip all gradients by the ratio of global_norm. Otherwise, all gradients remain as they are.
    # gradient_norm is the global_norm=sqrt(sum([l2norm(t)**2 for t in gradients]))
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    # compute and log the global_norm of the new clipped gradients
    gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))
    return clipped_gradients, gradient_norm_summary, gradient_norm


def run_batch_evaluation(model, session):
    """Compute evaluation metrics of the output of the given model.
      Args:
        model: model for running evaluation.
        sess: tensorflow session to use.
      Returns:
        The loss and the accuracy of the model outputs.
      """
    batch_count=0.0
    loss=0.0
    accuracy = 0.0
    while True:
        try:
            batch_loss,batch_accuracy,batch_size=model.eval(session)
            loss+=batch_loss
            accuracy+=batch_accuracy
            batch_count+=1
        except tf.errors.OutOfRangeError:
            break

    loss /= batch_count
    accuracy /= batch_count
    return loss, accuracy


def run_batch_prediction(model, session):
    """Run prediction of the given model.
          Args:
            model: model for computing predictions.
            sess: tensorflow session to use.
          Returns:
            The predictions of the model.
          """
    concat_predictions = None
    concat_input_sequence_length = None
    batch_count = 0
    while True:
        try:
            batch_count += 1
            predictions, batch_input_sequence_length = model.predict(session)
            if concat_predictions is None:
                concat_predictions = predictions
                concat_input_sequence_length = batch_input_sequence_length
            else:
                concat_predictions = np.append(concat_predictions, predictions, axis=0)
                concat_input_sequence_length = np.append(concat_input_sequence_length, predictions, axis=0)

        except tf.errors.OutOfRangeError:
            break
    return concat_predictions, concat_input_sequence_length



def load_model(model, session, name, ckpt):
    # Load a trained model from a checkpoint.
    start_time=time.time()
    #initialize all read-only tables of the graph, e.g., vocabulary tables or embedding tables.
    session.run(tf.local_variables_initializer())
    session.run(tf.tables_initializer())
    model.saver.restore(session, ckpt)
    print("loaded %s model parameters from %s, time %.2fs" % (name, ckpt, time.time()-start_time))
    return model



def create_or_load_model(model, session, name, model_dir, input_emb_weights=None):
    """Create a new model and initialize in the given session or load parameters in the given session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, session, name, latest_ckpt)
    else:
        start_time = time.time()
        #initialize all global and local variables in the graph, e.g., the model's weights.
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        # initialize all read-only tables of the graph, e.g., vocabulary tables or embedding tables.
        session.run(tf.tables_initializer())
        if input_emb_weights is not None:
            session.run(model.input_emb_init, feed_dict={model.input_emb_placeholder: input_emb_weights})
            print ("created model %s with new parameters, time %.2fs" %(name,time.time()-start_time))
    return model



def get_max_time(tensor):
    return tensor.shape[1].value or tf.shape(tensor)[1]


def add_summary(summary_writer, tag, value):
    """Add a new summary to the current summary_writer."""
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    # global_step value to record with the summary (optional).
    summary_writer.add_summary(summary, global_step=None)