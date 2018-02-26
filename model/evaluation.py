import tensorflow as tf
import codecs
import model_helper
import utils

def load_data(inference_input_file):
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(inference_input_file,mode="rb")) as f:
        inference_data = f.read().splitlines()

    return inference_data


def prediction(input_eval_file, output_eval_file, eval_output_folder, hparams, ckpt):
    eval_model = model_helper.create_eval_model()
    eval_sess = tf.Session(config=utils.get_config_proto(), graph=eval_model.graph)
    with eval_model.graph.as_default():
        loaded_eval_model = model_helper.load_model(eval_model.model, "prediction", eval_sess, ckpt)
    eval_sess.run(eval_model.iterator.initializer, feed_dict = {
        eval_model.input_file_placeholder: input_eval_file,
        eval_model.output_file_placeholder: output_eval_file
    })
    test_loss = model_helper.compute_loss(loaded_eval_model, eval_sess)
    print("Test loss: %.3f"%test_loss)

