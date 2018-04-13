import tensorflow as tf
import model_helper


class RNN(object):

    def __init__(self, hparams, mode, iterator, input_vocab_table=None):
        self.n_classes = hparams.n_classes
        self.vocab_size = hparams.vocab_size
        self.input_sequence_length = iterator.input_sequence_length
        self.mode = mode
        self.inputs = iterator.input
        self.targets = iterator.target
        self.input_vocab_table = input_vocab_table
        self.batch_size = iterator.batch_size

        #Initializer for all model parameters.
        initializer = tf.random_uniform_initializer(-hparams.init_weight,hparams.init_weight,seed=hparams.random_seed)
        tf.get_variable_scope().set_initializer(initializer)
        # Create embedding layer.
        self.input_embedding, self.input_emb_init, self.input_emb_placeholder = model_helper.create_embeddings\
                                      (vocab_size=self.vocab_size,
                                       emb_size=hparams.input_emb_size,
                                       emb_trainable=hparams.input_emb_trainable,
                                       emb_pretrain=hparams.input_emb_pretrain)

        # build graph of rnn model.
        res = self.build_graph(hparams)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss=res[1]
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            # Generate predictions (for INFER and EVAL mode only)
            self.logits = res[0]
            self.predictions = tf.nn.softmax(self.logits)
        ## Learning rate
        print("  start_decay_step=%d, learning_rate=%g, decay_steps %d,"
                  " decay_factor %g" % (hparams.start_decay_step, hparams.learning_rate,
                                       hparams.decay_steps, hparams.decay_factor))
        self.global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        # Gradients and sgd update operation for model training.
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            # Optimizer
            if hparams.optimizer == "sgd":
                # perform SGD with a learning rate with exponential decay
                self.learning_rate = tf.cond(
                    self.global_step < hparams.start_decay_step,
                    lambda: tf.constant(hparams.learning_rate),
                    lambda: tf.train.exponential_decay(
                        hparams.learning_rate,
                        (self.global_step - hparams.start_decay_step),
                        hparams.decay_steps,
                        hparams.decay_factor,
                        staircase=True),
                    name="learning_rate")
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr",self.learning_rate)
            elif hparams.optimizer == "adam":
                self.learning_rate = tf.constant(hparams.learning_rate)
                opt=tf.train.AdamOptimizer(self.learning_rate)
            # compute the gradients of train_loss w.r.t to the model trainable parameters.
            # if colocate_gradients_with_ops is true, the gradients will be computed in the same gpu/cpu device with the
            # original (forward-pass) operator
            gradients = tf.gradients(self.train_loss, params, colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)
            # clip gradients below a threshold to avoid explosion
            clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(gradients, max_gradient_norm=hparams.max_gradient_norm)
            self.grad_norm=grad_norm
            # ask the optimizer to apply the processed gradients. We give as argument a list of pairs (gradient,variable).
            self.update = opt.apply_gradients(
                zip(clipped_grads, params), global_step=self.global_step
            )
            self.train_summary = tf.summary.merge([
                tf.summary.scalar("lr", self.learning_rate),
                tf.summary.scalar("train_loss", self.train_loss),] + grad_norm_summary
            )
        # Calculate accuracy metric.
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.logits = res[0]
            correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.logits), len(self.logits.get_shape())-1), tf.cast(self.targets,tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Saver. As argument, we give the variables that are going to be saved and restored.
        # The Saver op will save the variables of the graph within it is defined. All graphs (train/eval/predict)
        # have a Saver operator.
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        # Print trainable variables
        print("# Trainable variables")
        for param in params:
            print("  %s, %s" % (param.name, str(param.get_shape())))
        import numpy as np
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total number of parameters: %d"%total_params)


    def build_graph(self, hparams):
        print ("Creating %s graph"%self.mode)
        dtype = tf.float32
        with tf.variable_scope("rnn_model",dtype=dtype):
            rnn_outputs, last_hidden_sate = self.build_rnn(hparams)
            # Unnormalized model outputs (before softmax!)
            logits=self.build_output_layer(hparams, rnn_outputs)
            # compute loss
            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                loss = None
            else:
                loss = self.compute_loss(logits)
        return logits, loss


    def build_output_layer(self, hparams, rnn_outputs):
        with tf.variable_scope("output_layer"):
            out_layer = tf.layers.Dense(hparams.n_classes, use_bias=hparams.out_bias, name="output_layer")
            logits = out_layer(rnn_outputs)
        return logits


    def build_rnn(self, hparams):
        # Look up embedding: emb_imp.shape = [batch_size, max_seq_length, num_units]
        emb_inp = tf.nn.embedding_lookup(self.input_embedding, self.inputs)
        # rnn_outputs.shape = [batch_size, max_seq_length, num_units]
        with tf.variable_scope("rnn") as scope:
            dtype=scope.dtype
            cell = model_helper.create_rnn_cell(hparams.unit_type, hparams.num_units, hparams.num_layers,
                                                    hparams.forget_bias, hparams.in_to_hidden_dropout, self.mode)
            # last_hidden_sate --> a Tensor of shape [batch_size, num_units] or a list of such Tensors for many layers
            # rnn_outputs --> a Tensor of shape [batch_size, max_seq_length, num_units].
            # sequence_length: It is used to zero-out outputs when past a batch element's true sequence length.
            rnn_outputs, last_hidden_sate = tf.nn.dynamic_rnn(cell, emb_inp,
                                                         dtype=dtype,
                                                         sequence_length=self.input_sequence_length)
        return rnn_outputs, last_hidden_sate



    def compute_loss(self, logits):
        target_output = self.targets
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(self.input_sequence_length, target_output.shape[1].value, dtype=logits.dtype)
        loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
        return loss


    def train(self, sess, options=None, run_metadata=None):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                        self.train_loss,
                        self.train_summary,
                        self.global_step,
                        self.learning_rate,
                        self.batch_size,
                        self.accuracy],
                        options=options,
                        run_metadata=run_metadata
                        )

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,self.accuracy,self.batch_size])


    def predict(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run([self.predictions, self.input_sequence_length])