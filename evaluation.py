import tensorflow as tf
import model_helper
import model
import numpy as np
import utils

def eval(model, sess, iterator, iterator_feed_dict):
    # initialize the iterator with the data on which we will evaluate the model.
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    loss, accuracy = model_helper.run_batch_evaluation(model, sess)
    return loss, accuracy


def predict(model, sess, iterator, iterator_feed_dict):
    # initialize the iterator with the data on which we will perform predictions.
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    predictions , input_sequence_length= model_helper.run_batch_prediction(model, sess)
    return predictions, input_sequence_length


def evaluate(hparams, ckpt):
    if hparams.model_architecture == "simple_rnn": model_creator = model.RNN
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
        eval_loss, eval_accuracy = eval(loaded_eval_model, eval_sess, eval_model.iterator, iterator_feed_dict)
        print("Eval loss: %.3f, Eval accuracy: %.3f"%(eval_loss,eval_accuracy))
    print("Starting predictions:")

    prediction_model = model_helper.create_infer_model(model_creator, hparams, tf.contrib.learn.ModeKeys.INFER)
    prediction_sess = tf.Session(config=utils.get_config_proto(), graph=prediction_model.graph)
    with prediction_model.graph.as_default():
        loaded_prediction_model = model_helper.load_model(prediction_model.model, prediction_sess, "prediction", ckpt)
        iterator_feed_dict = {
            prediction_model.input_file_placeholder: hparams.eval_input_path,
        }
    predictions, input_sequence_length=predict(loaded_prediction_model, prediction_sess, prediction_model.iterator, iterator_feed_dict)
    print("Saving predictions:")
    labels = np.argmax(predictions, axis=predictions.shape[-1]-1)
    with tf.gfile.GFile(hparams.eval_output_folder + "/" + hparams.predictions_filename, mode="w") as file:
        newLine=""
        for i,length in enumerate(input_sequence_length):
            label_seq=labels[i][:length]
            file.write(newLine+" ".join([str(l) for l in label_seq]))
            newLine="\n"

