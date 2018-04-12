import tensorflow as tf
import codecs
import os
import json


def maybe_parse_standard_hparams(hparams, hparams_path):
  """Override hparams values with existing standard hparams config."""
  if not hparams_path:
    return hparams

  if tf.gfile.Exists(hparams_path):
    print("# Loading standard hparams from %s" % hparams_path)
    with tf.gfile.GFile(hparams_path, "r") as f:
      hparams.parse_json(f.read())

  return hparams


def save_hparams(out_dir, hparams):
  """Save hparams."""
  hparams_file = os.path.join(out_dir, "hparams")
  print("  saving hparams to %s" % hparams_file)
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())


def load_hparams(model_dir):
  """Load hparams from an existing model directory."""
  hparams_file = os.path.join(model_dir, "hparams")
  if tf.gfile.Exists(hparams_file):
    print("# Loading hparams from %s" % hparams_file)
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print("  can't load hparams file")
        return None
    return hparams
  else:
    return None


def print_hparams(hparams, skip_patterns=None):
  """Print hparams, can skip keys based on pattern."""
  values = hparams.values()
  for key in sorted(values.keys()):
    if not skip_patterns or all(
        [skip_pattern not in key for skip_pattern in skip_patterns]):
      print("  %s=%s" % (key, str(values[key])))


def get_config_proto(log_device_placement=False, allow_soft_placement=True):
    config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                  allow_soft_placement = allow_soft_placement)
    # allocate as much GPU memory as is needed, based on runtime allocations.
    config_proto.gpu_options.allow_growth = True
    return config_proto
