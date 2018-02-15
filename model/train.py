import tensorflow as tf
import time
import os
import utils
import model_helper
import model
import numpy as np

def train(hparams):
    num_epochs = hparams.num_epochs
    num_ckpt_epochs = hparams.num_ckpt_epochs
    summary_name="train_log"
    out_dir = hparams.out_dir
    model_dir = out_dir
    log_device_placement = hparams.log_device_placement

    input_emb_weights = np.loadtxt(hparams.input_emb_file, delimiter=' ') if hparams.input_emb_file else None
    if hparams.model_architecture == "rnn-model": model_creator = model.RNN
    else: raise ValueError("Unknown model architecture. Only simple_rnn is supported so far.")
    #create 3  models in 3 graphs for train, evaluation and inference, with 3 sessions sharing the same variables.
    train_model = model_helper.create_train_eval_model(model_creator, hparams, hparams.train_input_path,
                                                       hparams.train_target_path, mode=tf.contrib.learn.ModeKeys.TRAIN)
    eval_model = model_helper.create_train_eval_model(model_creator, hparams, hparams.val_input_path,
                                                      hparams.val_target_path, tf.contrib.learn.ModeKeys.EVAL)
    infer_model = model_helper.create_infer_model(model_creator, hparams)

    # some configuration of gpus logging
    config_proto = utils.get_config_proto(log_device_placement=log_device_placement, allow_soft_placement=True)
    # create three separate sessions for trai/eval/infer
    train_sess=tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess=tf.Session(config=config_proto, graph=eval_model.graph)
    infer_sess=tf.Session(config=config_proto, graph=infer_model.graph)

    # create a new train model by initializing all graph variables
    # or load a model from the latest checkpoint within the model_dir.
    with train_model.graph.as_default():
        loaded_train_model= model_helper.create_or_load_model(train_model.model, train_sess, "train", model_dir, input_emb_weights)

    # create a log file with name summary_name in out_dir. The file is written asynchronously during the training process.
    summary_writer = tf.summary.FileWriter(os.path.join(out_dir,summary_name),train_model.graph)

    #run first evaluation before starting training
    val_loss = run_evaluation(eval_model, eval_sess, model_dir, summary_writer)
    print("Epoch 0: dev los: %.3f" % val_loss)

    # Start training
    start_train_time=time.time()
    epoch_time=0.0
    batch_loss, epoch_loss=0.0, 0.0
    batch_count=0.0

    #initialize train iterator
    train_sess.run(train_model.iterator.initializer)
    #keep lists of train/val losses for all epochs
    train_losses=[]
    val_losses=[]
    #train the model for num_epochs. One epoch means a pass through the whole train dataset, i.e., through all the batches.
    for epoch in range(num_epochs):
        #go through all batches for the current epoch
        while True:
            start_batch_time=0.0
            try:
                step_result = loaded_train_model.train(train_sess)
                (batch_loss, batch_summary)=step_result
                epoch_time += (time.time()-start_batch_time)
                epoch_loss += batch_loss
                batch_count += 1
            except tf.errors.OutOfRangeError:
                #when the iterator of the train batches reaches the end, break the loop
                #and reinitialize the iterator to start from the beginning of the train data.
                train_sess.run(train_model.iterator.initializer)
                break
        # average epoch loss and epoch time over batches
        epoch_loss /= batch_count
        epoch_time /= batch_count
        #print results if the current epoch is a print results epoch
        if (epoch +1) % num_ckpt_epochs ==0:
            print("Saving checkpoint: ")
            model_helper.add_summary(summary_writer, "train_loss", epoch_loss)
            # save checkpoint. global_step parameter is optional and is appended to the name of the checkpoint.
            loaded_train_model.saver.save(train_sess, os.path.join(out_dir, "rnn.ckpt"), global_step=epoch)

            print("Validation results: ")
            val_loss = run_evaluation(eval_model, eval_sess, model_dir, summary_writer)
            print(" epoch %d lr %g "
                  "train_loss %.3f, val_loss %.3f" %
                  (epoch, loaded_train_model.learning_rate.eval(session=train_sess),epoch_loss,val_loss))
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)

    # save final model
    loaded_train_model.saver.save(train_sess,os.path.join(out_dir,"rnn.ckpt"), global_step=num_epochs)
    print("Done training in %.2fK" % time.time() - start_train_time )
    min_val_loss = np.min(val_losses)
    min_val_idx = np.argmin(val_losses)
    print("Min val loss: %.3f at epoch %d"%(min_val_loss,min_val_idx[0]))
    summary_writer.close()


def run_evaluation(eval_model, eval_sess, model_dir, summary_writer):
    with eval_model.graph.as_default():
        loaded_eval_model = model_helper.create_or_load_model(eval_model.model, model_dir, eval_sess, "eval")
    eval_sess.run(eval_model.iterator.initializer)
    val_loss = model_helper.compute_loss(loaded_eval_model, eval_sess)
    model_helper.add_summary(summary_writer, "val_loss", val_loss)
    return val_loss


