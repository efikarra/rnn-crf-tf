import tensorflow as tf
import time
import os
import utils
import model_helper
import model
import numpy as np
import evaluation
from tensorflow.python.client import timeline


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
    #create 2  models in 2 graphs for train and evaluation, with 2 sessions sharing the same variables.
    train_model = model_helper.create_train_model(model_creator, hparams, hparams.train_input_path,
                                                  hparams.train_target_path, mode=tf.contrib.learn.ModeKeys.TRAIN)
    eval_model = model_helper.create_eval_model(model_creator, hparams, tf.contrib.learn.ModeKeys.EVAL)

    # some configuration of gpus logging
    config_proto = utils.get_config_proto(log_device_placement=log_device_placement, allow_soft_placement=True)
    # create two separate sessions for trai/eval
    train_sess=tf.Session(config=config_proto, graph=train_model.graph)
    eval_sess=tf.Session(config=config_proto, graph=eval_model.graph)

    # create a new train model by initializing all variables of the train graph in the train_sess.
    # or, using the latest checkpoint in the model_dir, load all variables of the train graph in the train_sess.
    # Note that at this point, the eval graph variables are not initialized.
    with train_model.graph.as_default():
        loaded_train_model = model_helper.create_or_load_model(train_model.model, train_sess, "train", model_dir, input_emb_weights)
    # create a log file with name summary_name in out_dir. The file is written asynchronously during the training process.
    # We also passed the train graph in order to be able to display it in Tensorboard
    summary_writer = tf.summary.FileWriter(os.path.join(out_dir,summary_name),train_model.graph)

    #run first evaluation before starting training
    dev_loss = run_evaluation(eval_model, eval_sess, model_dir, hparams.val_input_path, hparams.val_target_path, input_emb_weights, summary_writer)
    train_loss = run_evaluation(eval_model, eval_sess, model_dir, hparams.train_input_path, hparams.train_target_path,
                              input_emb_weights, summary_writer)
    print("Dev loss before training: %.3f" % dev_loss)
    print("Train loss before training: %.3f" % train_loss)
    # Start training
    start_train_time=time.time()
    avg_batch_time=0.0
    batch_loss, epoch_loss=0.0, 0.0
    batch_count=0.0

    #initialize train iterator in train_sess
    train_sess.run(train_model.iterator.initializer)
    #keep lists of train/val losses for all epochs
    train_losses=[]
    dev_losses=[]

    # vars to compute timeline of operations
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    #train the model for num_epochs. One epoch means a pass through the whole train dataset, i.e., through all the batches.
    step=0
    for epoch in range(num_epochs):
        #go through all batches for the current epoch
        while True:
            start_batch_time = time.time()
            try:
                # this call will run operations of train graph in train_sess
                if step%10==0:
                    step_result = loaded_train_model.train(train_sess,options=options,run_metadata=run_metadata)
                    summary_writer.add_run_metadata(run_metadata, 'step%d' % step)
                    #compute pipeline
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    if hparams.timeline:
                        with open('timelines/timeline_02_step_%d.json' % step, 'w') as f:
                            f.write(chrome_trace)
                else: step_result = loaded_train_model.train(train_sess,options=None,run_metadata=None)

                (_, batch_loss, batch_summary, global_step, learning_rate, batch_size)=step_result
                avg_batch_time += (time.time()-start_batch_time)
                print(batch_loss)
                epoch_loss += batch_loss
                batch_count += 1
                step+=1
            except tf.errors.OutOfRangeError:
                #when the iterator of the train batches reaches the end, break the loop
                #and reinitialize the iterator to start from the beginning of the train data.
                train_sess.run(train_model.iterator.initializer)
                break
        # average epoch loss and epoch time over batches
        epoch_loss /= batch_count
        avg_batch_time /= batch_count
        print("Number of batches: %d"%batch_count)
        #print results if the current epoch is a print results epoch
        if (epoch +1) % num_ckpt_epochs == 0:
            print("Saving checkpoint...")
            model_helper.add_summary(summary_writer, "train_loss", epoch_loss)
            # save checkpoint. We save the values of the variables of the train graph.
            # train_sess is the session in which the train graph was launched.
            # global_step parameter is optional and is appended to the name of the checkpoint.
            loaded_train_model.saver.save(train_sess, os.path.join(out_dir, "rnn.ckpt"), global_step=epoch)

            print("Results: ")
            dev_loss = run_evaluation(eval_model, eval_sess, model_dir, hparams.val_input_path, hparams.val_target_path, input_emb_weights, summary_writer)
            # tr_loss = run_evaluation(eval_model, eval_sess, model_dir, hparams.train_input_path, hparams.train_target_path, input_emb_weights, summary_writer)
            # print("check %.3f:"%tr_loss)
            print(" epoch %d lr %g "
                  "train_loss %.3f, dev_loss %.3f avg_batch_time %f"%
                  (epoch, loaded_train_model.learning_rate.eval(session=train_sess), epoch_loss, dev_loss,
                                 avg_batch_time))
            train_losses.append(epoch_loss)
            dev_losses.append(dev_loss)
        batch_count = 0.0
        avg_batch_time = 0.0

    # save final model
    loaded_train_model.saver.save(train_sess,os.path.join(out_dir,"rnn.ckpt"), global_step=num_epochs)
    print("Done training in %.2fK" % (time.time() - start_train_time) )
    min_dev_loss = np.min(dev_losses)
    min_dev_idx = np.argmin(dev_losses)
    print("Min val loss: %.3f at epoch %d"%(min_dev_loss,min_dev_idx))
    summary_writer.close()


def run_evaluation(eval_model, eval_sess, model_dir, input_eval_file, output_eval_file, input_emb_weights, summary_writer):
    with eval_model.graph.as_default():
        # initialize the variables of the eval graph in eval_sess or load them from a checkpoint.
        loaded_eval_model = model_helper.create_or_load_model(eval_model.model, eval_sess, "eval", model_dir, input_emb_weights)
    eval_iterator_feed_dict = {
        eval_model.input_file_placeholder: input_eval_file,
        eval_model.output_file_placeholder: output_eval_file
    }
    dev_loss = evaluation.eval(loaded_eval_model, eval_sess, eval_model.iterator, eval_iterator_feed_dict)
    model_helper.add_summary(summary_writer, "dev_loss", dev_loss)
    return dev_loss


