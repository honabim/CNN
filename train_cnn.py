#! /usr/bin/env python
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
sys.path.insert(0, '/model_cnn')
from model_cnn import data_helpers,text_cnn

#import data_helpers
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .3, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("find_data_file", "./Data_CNN/Data.findRestaurantsByCity", "Data source for the findRestaurantsByCity data.")
tf.flags.DEFINE_string("greet_data_file", "./Data_CNN/Data.greet", "Data source for the greet data.")
tf.flags.DEFINE_string("bye_data_file", "./Data_CNN/Data.bye", "Data source for the bye data.")
tf.flags.DEFINE_string("affirmative_data_file", "./Data_CNN/Data.affirmative", "Data source for the affirmative data.")
tf.flags.DEFINE_string("negative_data_file", "./Data_CNN/Data.negative", "Data source for the negative data.")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 8, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.4, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 25, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 28, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every",28, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.find_data_file,FLAGS.greet_data_file,FLAGS.bye_data_file,FLAGS.affirmative_data_file,FLAGS.negative_data_file)
    #x_text, y = data_helpers.load_data_and_labels(FLAGS.greet_data_file, FLAGS.bye_data_file)
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    print("len(y=)",len(y))
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    print("-------------------------")
    print("shuffle_indices=",shuffle_indices)
    print("-------------------------")
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    print("dev_sample_index=",dev_sample_index)
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev
    
def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    generations=2800
    eval_every=28
    train_acc=[]
    test_acc=[]
    train_loss=[]
    test_loss=[]

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = text_cnn.TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            timestamp="model" # Tai's code
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "run_cnn", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
               
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                train_acc.append(accuracy)
                train_loss.append(loss)
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """

                print("---------in dev_step--------------------------")
                print("x_batch=",x_batch)
                print("y_batch=",y_batch)
                print("-----------------------------------------------")
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                test_acc.append(accuracy)
                test_loss.append(loss)
                print("-----------------------------")
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            print('type(batches)=',type(batches))
            m=0
            n=0
            for batch in batches:
                m=m+1
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("m=",m)
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                 
                    print("")
                # print("\nEvaluation:")
                # dev_step(x_dev, y_dev, writer=dev_summary_writer)
                # print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                # print("Saved model checkpoint to {}\n".format(path))
            print("m'=",m)
            print("len(train_acc)=",len(train_acc))
            print("len(test_acc)=",len(test_acc))
            print("len(train_loss)=",len(train_loss))
            train_acc1=[]
            train_loss1=[]
            for x in range(0,2800,28):
                train_acc1.append(train_acc[x])
                train_loss1.append(train_loss[x])
            import matplotlib.pyplot as plt
            eval_indices = range(0, generations, eval_every)

            # Plot loss over time
            plt.plot(eval_indices, train_loss1, 'k-', label='Train Set loss')
            plt.plot(eval_indices,test_loss,'r-', label='Test Set loss')
            plt.title('Train and Test Loss')
            plt.xlabel('Generation')
            plt.ylabel('Softmax Loss')
            plt.legend(loc='lower left')
            plt.show()

            # Plot train and test accuracy
            plt.plot(eval_indices, train_acc1, 'k-', label='Train Set Accuracy')
            plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
            plt.title('Train and Test Accuracy')
            plt.xlabel('Generation')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.show()

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()