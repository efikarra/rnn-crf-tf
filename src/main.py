import argparse
import tensorflow as tf
import os
import sys
import train
from utils import utils
import evaluation
from utils import vocab_utils
import embedding


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--train_input_path", type=str, default=None,
                        help="Train input file path.")
    parser.add_argument("--train_target_path", type=str, default=None,
                        help="Train target file path.")
    parser.add_argument("--val_input_path", type=str, default=None,
                        help="Validation input file path for validation dataset.")
    parser.add_argument("--val_target_path", type=str, default=None,
                        help="Validation target file path for validation dataset.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")
    parser.add_argument("--hparams_path", type=str, default=None,
                        help=("Path to hparams json file that overrides"
                              "hparams values from flags."))
    parser.add_argument("--input_emb_file", type=str, default=None, help="Input embedding external file.")
    parser.add_argument("--create_new_embeddings", type=bool, default=True, help="Whether to create new embeddings.")
    parser.add_argument("--embedding_path", type=str, default=None, help="Pretrained embeddings file path.")
    # Vocab
    parser.add_argument("--vocab_path", type=str, default=None, help="Vocabulary file path.")
    parser.add_argument("--unk", type=str, default="<unk>",
                        help="Symbol for the unknown token.")
    parser.add_argument("--pad", type=str, default="<pad>",
                        help="Symbol for the pad symbol.")

    # Input sequence max length
    parser.add_argument("--input_max_len", type=int, default=None,
                        help="Max length of input sequence.")

    # network
    parser.add_argument("--model_architecture", type=str, default="simple_rnn",
                        help="simple_rnn. Model architecture.")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help="Initial weights from [-init_weight, init_weight].")
    parser.add_argument("--num_units", type=int, default=32, help="Number of hidden units of rnn.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers.")
    parser.add_argument("--in_to_hidden_dropout", type=float, default=0.0, help="dropout.")
    parser.add_argument('--unit_type', type=str, default="rnn",
                        help="rnn | lstm | gru | layer_norm_lstm.")
    parser.add_argument("--n_classes", type=int, default=None, help="Number of output classes.")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--emb_size", type=int, default=32, help="Input embedding size.")
    parser.add_argument("--input_emb_trainable", type=bool, default=True,
                        help="Whether to train embedding layer weights.")
    parser.add_argument("--out_bias", type=bool, default=True, help="Whether to use bias at the output layer.")
    # training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--num_ckpt_epochs", type=int, default=2,
                        help="Number of epochs until the next checkpoint saving.")

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--start_decay_step", type=int, default=0,
                        help="When we start to decay")
    parser.add_argument("--decay_steps", type=int, default=10000,
                        help="How frequent we decay")
    parser.add_argument("--decay_factor", type=float, default=0.98,
                        help="How much we decay.")
    parser.add_argument("--colocate_gradients_with_ops", type="bool", nargs="?",
                        const=True,
                        default=True,
                        help=("Whether try colocating gradients with "
                              "corresponding op"))
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")

    # Other
    parser.add_argument("--gpu", type=int, default=0,
                        help="Gpu machine to run the code (if gpus available)")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--log_device_placement", type="bool", nargs="?",
                        const=True, default=False, help="Debug GPU allocation.")
    parser.add_argument("--timeline", type="bool", nargs="?",
                        const=True, default=False, help="Log timeline information.")

    # Evaluation/Prediction
    parser.add_argument("--eval_output_folder", type=str, default=None,
                        help="Output folder to save evaluation data.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint file of trained model.")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation mode.")
    parser.add_argument("--predict_batch_size", type=int, default=32,
                        help="Batch size for prediction mode.")
    parser.add_argument("--eval_input_path", type=str, default=None,
                        help="Input file path to perform evaluation and/or prediction.")
    parser.add_argument("--eval_target_path", type=str, default=None,
                        help="Output file path to perform evaluation and prediction.")
    parser.add_argument("--predictions_filename", type=str, default="predictions.txt",
                        help="Filename to save predictions.")


def create_hparams(flags):
    return tf.contrib.training.HParams(
        # data
        input_emb_file=flags.input_emb_file,
        out_dir=flags.out_dir,
        train_input_path=flags.train_input_path,
        train_target_path=flags.train_target_path,
        val_input_path=flags.val_input_path,
        val_target_path=flags.val_target_path,
        hparams_path=flags.hparams_path,
        # Vocab
        vocab_path=flags.vocab_path,
        unk=flags.unk,
        pad=flags.pad,
        # Input sequence max length
        input_max_len=flags.input_max_len,
        # network
        model_architecture=flags.model_architecture,
        num_units=flags.num_units,
        init_weight=flags.init_weight,
        num_layers=flags.num_layers,
        in_to_hidden_dropout=flags.in_to_hidden_dropout,
        n_classes=flags.n_classes,
        forget_bias=flags.forget_bias,
        unit_type=flags.unit_type,
        input_emb_size=flags.emb_size,
        input_emb_trainable=flags.input_emb_trainable,
        create_new_embeddings=flags.create_new_embeddings,
        embedding_path=flags.embedding_path,
        out_bias=flags.out_bias,
        # training
        batch_size=flags.batch_size,
        num_epochs=flags.num_epochs,
        num_ckpt_epochs=flags.num_ckpt_epochs,
        # optimizer
        colocate_gradients_with_ops=flags.colocate_gradients_with_ops,
        learning_rate=flags.learning_rate,
        optimizer=flags.optimizer,
        start_decay_step=flags.start_decay_step,
        decay_steps=flags.decay_steps,
        decay_factor=flags.decay_factor,
        max_gradient_norm=flags.max_gradient_norm,
        # evaluation/prediction
        eval_output_folder=flags.eval_output_folder,
        ckpt=flags.ckpt,
        eval_input_path=flags.eval_input_path,
        eval_target_path=flags.eval_target_path,
        eval_batch_size=flags.eval_batch_size,
        predict_batch_size=flags.predict_batch_size,
        predictions_filename=flags.predictions_filename,
        # Other
        random_seed=flags.random_seed,
        log_device_placement=flags.log_device_placement,
        gpu=flags.gpu,
        timeline=flags.timeline
    )


def extend_hparams(hparams):
    """Extend training hparams."""
    hparams.add_hparam("input_emb_pretrain", hparams.input_emb_file is not None)
    # Check if vocab has the unk and pad symbols as first words. If not, create a new vocab file with these symbols as
    # the first two words.
    vocab_size, vocab_path = vocab_utils.check_vocab(hparams.vocab_path, hparams.out_dir,
                                                     unk=hparams.unk, pad=hparams.pad)
    vocab, _ = vocab_utils.load_vocab(vocab_path)
    # Generating embeddings if flag is true or file is not present
    if hparams.create_new_embeddings or os.path.isfile(hparams.input_emb_file) is False:
        embedding.save_embedding(vocab, hparams.embedding_path, hparams.input_emb_file)
    hparams.add_hparam("vocab_size", vocab_size)
    hparams.set_hparam("vocab_path", vocab_path)
    if not tf.gfile.Exists(hparams.out_dir):
        tf.gfile.MakeDirs(hparams.out_dir)
    return hparams


def process_or_load_hparams(out_dir, default_hparams, hparams_path):
    hparams = default_hparams
    # if a Hparams path is given as argument, override the default_hparams.
    hparams = utils.maybe_parse_standard_hparams(hparams, hparams_path)
    # extend HParams to add some parameters necessary for the training.
    hparams = extend_hparams(hparams)
    # Save HParams
    utils.save_hparams(out_dir, hparams)

    # Print HParams
    print("Print hyperparameters:")
    utils.print_hparams(hparams)
    return hparams


def run_main(default_hparams, train_fn, evaluation_fn):
    out_dir = default_hparams.out_dir
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)
    hparams = process_or_load_hparams(out_dir, default_hparams, default_hparams.hparams_path)

    # restrict tensoflow to run only in the specified gpu. This has no effect if run on a machine with no gpus. You dont care about gpus now.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu)
    # if there is an evaluation output folder in the hparams we proceed with evaluation based on an existing model.
    # Otherwise, we train a new model.
    if hparams.eval_output_folder:
        # Evaluation
        ckpt = hparams.ckpt
        # if no checkpoint path is given as input, load the latest checkpoint from the output folder.
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
        evaluation_fn(hparams, ckpt)
    else:
        # Train
        train_fn(hparams)


def main(flags):
    # create HParams object from flags parameter.
    # HParams is a class to hold a set of hyperparameters as name-value pairs.
    default_hparams = create_hparams(flags)
    train_fn = train.train
    evaluation_fn = evaluation.evaluate
    run_main(default_hparams, train_fn, evaluation_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # add the possible command line arguments to the parser.
    add_arguments(parser)
    # parse command line args
    flags, unparsed = parser.parse_known_args()
    tf.app.run(main=main(flags), argv=[sys.argv[0]] + unparsed)
