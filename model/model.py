import tensorflow as tf
import model_helper


class RNN(object):

    def __init__(self, hparams, mode, iterator, input_vocab_table=None, reverse_input_vocab_table = None):
        self.time_major=hparams.time_major
        self.n_classes = hparams.n_classes
        self.vocab_size = hparams.vocab_size
        self.input_sequence_length = iterator.input_sequence_length
        self.mode = mode
        self.inputs = iterator.input
        self.targets = iterator.target
        self.input_vocab_table = input_vocab_table
        self.reverse_input_vocab_table = reverse_input_vocab_table
        self.input_emb_pretrain = hparams.input_emb_pretrain
        self.batch_size = iterator.batch_size

        #Initializer
        initializer = tf.random_uniform_initializer(-hparams.init_weight,hparams.init_weight,seed=hparams.random_seed)
        tf.get_variable_scope().set_initializer(initializer)
        # Create embedding layer
        self.input_embedding, self.input_emb_init, self.input_emb_placeholder = model_helper.create_embeddings\
                                      (vocab_size=self.vocab_size,
                                       emb_size=hparams.input_emb_size,
                                       emb_trainable=hparams.input_emb_trainable,
                                       emb_pretrain=self.input_emb_pretrain)

        # build graph of main rnn model
        res = self.build_graph(hparams)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.train_loss = res[1]
        elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
            self.eval_loss=res[1]
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            # Generate predictions (for INFER and EVAL mode)
            self.logits = res[0]
            self.predictions = {
                    "classes": tf.argmax(input=tf.nn.softmax(self.logits), axis=1),
                    "probabilities": tf.nn.softmax(self.logits)
                }
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
        # Saver. As argument, we give the variables that are going to be saved and restored.
        # The Saver op will save the variables of the graph within it is defined. All graphs (train/eval/predict) have
        # have a Saver operator.
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        # print trainable params
        # Print trainable variables
        print("# Trainable variables")
        for param in params:
            print("  %s, %s" % (param.name, str(param.get_shape())))
        import numpy as np
        total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print(total_params)


    def build_graph(self, hparams):
        print ("Creating %s graph"%self.mode)
        dtype = tf.float32
        with tf.variable_scope("rnn_model",dtype=dtype):
            rnn_last_states=self._build_rnn(hparams)
            logits=self._build_output_layer(hparams, rnn_last_states)
            # compute loss
            if self.mode == tf.contrib.learn.ModeKeys.INFER:
                loss = None
            else:
                loss = self.compute_loss(logits)
        return logits, loss


    def _build_output_layer(self, hparams, rnn_outputs):
        with tf.variable_scope("output_layer"):
            out_layer = tf.layers.Dense(hparams.n_classes, use_bias=False, name="output_layer")
            logits = out_layer(rnn_outputs)
        return logits



    def _build_rnn(self, hparams):
        if self.time_major:
            self.inputs=tf.transpose(self.inputs)

        emb_inp = tf.nn.embedding_lookup(self.input_embedding, self.inputs)
        last_hidden_sate =[]
        # RNN outputs: [max_time, batch_size, num_units]
        with tf.variable_scope("rnn") as scope:
            dtype=scope.dtype
            # Look up embedding, emb_imp: [max_time, batch_size, num_units]
            if hparams.rnn_type == "uni":
                cell = model_helper.create_rnn_cell(hparams.unit_type, hparams.num_units, hparams.num_layers,
                                                    hparams.forget_bias, hparams.in_to_hidden_dropout, self.mode)
                # encoder_state --> a Tensor of shape `[batch_size, cell.state_size]` or a list of such Tensors for many layers
                _, last_hidden_sate = tf.nn.dynamic_rnn(cell, emb_inp,
                                                         dtype=dtype,
                                                         sequence_length=self.input_sequence_length,
                                                         time_major=self.time_major)
            elif hparams.rnn_type == "bi":
                num_bi_layers = int(hparams.num_layers / 2)
                print("num_bi_layers %d"%num_bi_layers)
                _, bi_last_hidden_state=self._build_bidirectional_rnn(emb_inp,dtype,hparams,num_bi_layers)
                # if the encoder has 1 layer per bi-rnn, it means that it has 1 fwd and 1 bwd layers -> in total it has 2 layers.
                # and every fwd and bwd layer has enc_units each -> in total 2*enc_units
                if num_bi_layers == 1:
                    last_hidden_sate = bi_last_hidden_state
                else:
                    # alternatively concat forward and backward states
                    last_hidden_sate = []
                    for layer_id in range(num_bi_layers):
                        # bi_encoder_state[0] are all the enc_layers/2 fwd states.
                        last_hidden_sate.append(bi_last_hidden_state[0][layer_id])  # forward
                        # bi_encoder_state[1] are all the enc_layers/2 bwd states.
                        last_hidden_sate.append(bi_last_hidden_state[1][layer_id])  # backward
                        last_hidden_sate = tuple(last_hidden_sate)
            else:
                raise ValueError("Unknown rnn type: %s" % hparams.rnn_type)
        return last_hidden_sate

    def _build_bidirectional_rnn(self,inputs,dtype,hparams,num_bi_layers):
        # Construct forward and backward cells.
        #each one has num_bi_layers layers. Each layer has num_units.
        fw_cell = model_helper.create_rnn_cell(hparams.unit_type, hparams.num_units, num_bi_layers,
                                                    hparams.forget_bias, hparams.in_to_hidden_dropout, self.mode)
        bw_cell = model_helper.create_rnn_cell(hparams.unit_type, hparams.num_units, num_bi_layers,
                                                    hparams.forget_bias, hparams.in_to_hidden_dropout, self.mode)

        # initial_state_fw, initial_state_bw are initialized to 0
        # bi_outputs is a tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor
        # bi_state is a tuple (output_state_fw, output_state_bw) with the forward and the backward final states.
        # Each state has num_units.
        # num_bi_layers>1, we have a list of num_bi_layers tuples.
        bi_outputs,bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=self.input_sequence_length,
            time_major=self.time_major)

        # return fw and bw outputs,i.e., ([h1_fw;h1_bw],...,[hT_fw;hT_bw]) concatenated.
        return tf.concat(bi_outputs,-1),bi_state


    def compute_loss(self, logits):
        target_output = self.targets
        if self.time_major:
            target_output = tf.transpose(target_output)
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        loss = tf.reduce_sum(crossent)/tf.to_float(self.batch_size)
        return loss


    def train(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return sess.run([self.update,
                        self.train_loss,
                        self.train_summary,
                        self.global_step,
                        self.learning_rate,
                        self.batch_size,tf.transpose(self.inputs), self.targets])

    def eval(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return sess.run([self.eval_loss,self.batch_size])


    def predict(self, sess):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return sess.run(self.predictions)