import tensorflow as tf
import codecs
import model_helper

def load_data(inference_input_file):
    with codecs.getreader("utf-8")(
            tf.gfile.GFile(inference_input_file,mode="rb")) as f:
        inference_data = f.read().splitlines()

    return inference_data


def inference(infer_model, infer_sess, inference_input_file, inference_output_folder, ckpt):
    infer_data = load_data(inference_input_file)

    with infer_model.graph.as_default():
        loaded_infer_model = model_helper.load_model(infer_model.model, ckpt, infer_sess, "infer")
    infer_sess.run(infer_model.iterator.initializer)
    loaded_infer_model.infer(infer_sess)