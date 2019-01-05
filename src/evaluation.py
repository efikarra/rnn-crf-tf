import tensorflow as tf
import model_helper
import model
import numpy as np
from utils import utils
import cPickle
import os

def eval(model, sess, iterator, iterator_feed_dict):
    # initialize the iterator with the data on which we will evaluate the model.
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    loss, accuracy = model_helper.run_batch_evaluation(model, sess)
    return loss, accuracy



def eval_and_precit(model, sess, iterator, iterator_feed_dict):
    # initialize the iterator with the data on which we will evaluate the model.
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    loss, accuracy, predictions = model_helper.run_batch_evaluation_and_prediction(model, sess)
    return loss, accuracy, predictions

def evaluate(hparams, ckpt):
    if hparams.model_architecture == "simple_rnn":
        model_creator = model.RNN
    else:
        raise ValueError("Unknown model architecture. Only simple_rnn is supported so far.")
    print("Starting evaluation and predictions:")
    # create eval graph.
    eval_model = model_helper.create_eval_model(model_creator, hparams, tf.contrib.learn.ModeKeys.EVAL)
    eval_sess = tf.Session(config=utils.get_config_proto(), graph=eval_model.graph)
    with eval_model.graph.as_default():
        # load pretrained model.
        loaded_eval_model = model_helper.load_model(eval_model.model, eval_sess, "evaluation", ckpt)
    iterator_feed_dict = {
        eval_model.input_file_placeholder: hparams.eval_input_path,
        eval_model.output_file_placeholder: hparams.eval_target_path
    }
    eval_loss, eval_accuracy, predictions = eval_and_precit(loaded_eval_model, eval_sess, eval_model.iterator,
                                                            iterator_feed_dict)
    print("Eval loss: %.3f, Eval accuracy: %.3f" % (eval_loss, eval_accuracy))
    # only models with CRF include trans. params.
    transition_params = eval_sess.run(loaded_eval_model.transition_params)
    if transition_params is not None:
        print("Saving transition parameters:")
        np.savetxt(os.path.join(hparams.eval_output_folder, "transition_params.txt"), transition_params)

    print("Saving predictions:")
    cPickle.dump(predictions,
                 open(os.path.join(hparams.eval_output_folder,
                                   hparams.predictions_filename.split(".")[0] + ".pickle"),
                      "wb"))