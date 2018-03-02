import tensorflow as tf
import model_helper
import utils
import model
import io

def eval(model, sess, iterator, iterator_feed_dict):
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    loss = model_helper.run_batch_evaluation(model, sess)
    return loss


def predict(model, sess, iterator, iterator_feed_dict):
    sess.run(iterator.initializer, feed_dict=iterator_feed_dict)
    predictions=model.predict(sess)
    return predictions

def save_probabilities(probabilities, filepath):
    with io.open(filepath, 'w', encoding='utf-8') as file:
        for probs in probabilities:
            res = ""
            for i, ps in enumerate(probs):
                res = " ".join([str(p) for p in ps])
                if i < len(probs) - 1:
                    res += ","
                file.write(res)
            file.write("\n")

def save_labels(labels, filepath):
    with io.open(filepath, 'w', encoding='utf-8') as file:
        newline = ""
        for label in labels:
            file.write(newline+str(label))
            newline="\n"

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
    prediction_model = model_helper.create_infer_model(model_creator, hparams, tf.contrib.learn.ModeKeys.INFER)
    prediction_sess = tf.Session(config=utils.get_config_proto(), graph=prediction_model.graph)
    with prediction_model.graph.as_default():
        loaded_prediction_model = model_helper.load_model(prediction_model.model, prediction_sess, "prediction", ckpt)
        iterator_feed_dict = {
            prediction_model.input_file_placeholder: hparams.val_input_path,
        }
        predictions=predict(loaded_prediction_model, prediction_sess, prediction_model.iterator, iterator_feed_dict)
        print(predictions["probabilities"][0:2])
        print(predictions["classes"][0:2])


