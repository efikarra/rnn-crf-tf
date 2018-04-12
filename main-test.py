import main
import argparse




if __name__ == '__main__':
    params = argparse.Namespace()

    # Set model parameters.

    #Input data parameters.
    params.out_dir='experiments/out_model'

    # Add your data filepaths here.
    # Train data files.
    params.train_input_path='experiments/data/example_input.txt'
    params.train_target_path='experiments/data/example_target.txt'
    # Validation data files.
    params.val_input_path='experiments/data/example_input.txt'
    params.val_target_path='experiments/data/example_target.txt'

    # Let this None. Input file for pretrained embeddings. You don't have to add pretrained embeddings now.
    params.input_emb_file = None
    # Hparams filepath if you want to parse hyperparameters from an external file.
    params.hparams_path = None

    # Vocab parameters. Add your vocabulary filepath here.
    # Vocab file does not have to contain the unk and pad symbols.
    # The code will check if unk and pad symbols are the first two words of vocab. If not, they will be added
    # and an extended vocab will be saved in params.out_dir.
    params.vocab_path='experiments/data/vocab_test.txt'
    # What symbols to use for unk and pad.
    params.unk='<unk>'
    params.pad='<pad>'
    # Input sequence max length.
    params.input_max_len=None

    # network
    params.model_architecture = 'simple_rnn'
    params.num_units=64
    params.init_weight=0.1
    params.num_layers=1
    params.in_to_hidden_dropout=0.1
    # change this based on the number of target classes of your dataset.
    params.n_classes=3
    params.forget_bias=1.0
    params.unit_type='rnn'
    params.emb_size=100
    params.input_emb_trainable = True
    params.out_bias=True
    # training
    params.batch_size=64
    params.num_epochs=10
    params.num_ckpt_epochs=1
    # optimizer
    params.learning_rate=0.01
    params.optimizer='adam'
    params.colocate_gradients_with_ops = True
    params.start_decay_step = 0
    params.decay_steps = 10000
    params.decay_factor = 0.98
    params.max_gradient_norm = 5.0
    #Other
    # you don't care at all about the following 4 parameters.
    params.gpu = None
    params.random_seed = None
    params.log_device_placement = False
    params.timeline = False

    # output folder to save evaluation results. Let this None to train a model.
    # If you want to perform evaluation, set a value for params.eval_output_folder such as 'experiments/out_eval'
    params.eval_output_folder=None
    params.eval_output_folder = 'experiments/out_eval'
    # Checkpoint filepath to load a trained model.
    params.ckpt =None
    # params.ckpt='experiments/out_model/<CHECKPOINT_NAME>'

    # Test data files to run evaluation on.
    params.eval_input_path='experiments/data/example_input.txt'
    params.eval_target_path='experiments/data/example_target.txt'
    params.eval_batch_size=64
    params.predict_batch_size=64
    #filename to save predictions on the test set. They will be saved in the eval_output_folder.
    params.predictions_filename='predictions.txt'

    main.main(params)
