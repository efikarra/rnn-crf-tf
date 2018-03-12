import tensorflow as tf
import model_helper
import model
import os
import numpy as np
import utils


def eval(model, sess, iterator, iterator_feed_dict):
    # initialize the iterator with the data on which we will evaluate the model.
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    loss = model_helper.run_batch_evaluation(model, sess)
    return loss


def predict(model, sess, iterator, iterator_feed_dict):
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    concat_predictions = {}
    batch_count = 0
    while True:
        try:
            batch_count += 1
            predictions = model.predict(sess)
            if "probabilities" not in concat_predictions:
                concat_predictions["probabilities"]=predictions["probabilities"]
            else: concat_predictions["probabilities"]=np.append(concat_predictions["probabilities"], predictions["probabilities"], axis=0)
            if "classes" not in concat_predictions:
                concat_predictions["classes"]=predictions["classes"]
            else: concat_predictions["classes"]=np.append(concat_predictions["classes"],predictions["classes"], axis=0)

        except tf.errors.OutOfRangeError:
            break
    return concat_predictions


def evaluate(hparams, ckpt):
    if hparams.model_architecture == "rnn-model": model_creator = model.RNN
    else: raise ValueError("Unknown model architecture. Only simple_rnn is supported so far.")

    if hparams.val_target_path:
        eval_model = model_helper.create_eval_model(model_creator, hparams, tf.contrib.learn.ModeKeys.EVAL)
        eval_sess = tf.Session(config=utils.get_config_proto(), graph=eval_model.graph)
        with eval_model.graph.as_default():
            loaded_eval_model = model_helper.load_model(eval_model.model, eval_sess, "evaluation", ckpt)
        iterator_feed_dict={
            eval_model.input_file_placeholder: hparams.eval_input_path,
            eval_model.output_file_placeholder: hparams.eval_target_path
        }
        eval_loss = eval(loaded_eval_model, eval_sess, eval_model.iterator, iterator_feed_dict)
        print("Eval loss: %.3f"%eval_loss)
    print("Starting predictions:")

    prediction_model = model_helper.create_infer_model(model_creator, hparams, tf.contrib.learn.ModeKeys.INFER)
    prediction_sess = tf.Session(config=utils.get_config_proto(), graph=prediction_model.graph)
    with prediction_model.graph.as_default():
        loaded_prediction_model = model_helper.load_model(prediction_model.model, prediction_sess, "prediction", ckpt)
        iterator_feed_dict = {
            prediction_model.input_file_placeholder: hparams.val_input_path,
        }
    predictions=predict(loaded_prediction_model, prediction_sess, prediction_model.iterator, iterator_feed_dict)
    np.savetxt(os.path.join(hparams.eval_output_folder, "classes.txt"), predictions["classes"])
    np.savetxt(os.path.join(hparams.eval_output_folder, "probabilities.txt"), predictions["probabilities"])
    # save_labels(predictions["classes"], os.path.join(hparams.eval_output_folder, "classes"))
    # save_probabilities(predictions["probabilities"], os.path.join(hparams.eval_output_folder, "probabilities"))


