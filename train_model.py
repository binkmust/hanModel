#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from HAN_model_shb import HAN
from data_helper_shb import iter_batch, load_dataset
import numpy as np
from sklearn import metrics
import sys

import time

tf.flags.DEFINE_string("yelp_json_path", '../data/TitleOutput.txt', "data directory")
tf.flags.DEFINE_integer("vocab_size", 33692, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 34, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 300, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 200, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_sent_in_doc", 1, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_word_in_sent", 50, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")


FLAGS = tf.flags.FLAGS

def evaluate(sess,val_x,va_y):
    val_len = len(val_x)
    total_loss = 0.0
    total_acc = 0.0
    for x_, y_ in iter_batch(val_x, val_y, FLAGS.batch_size):
        feed_dict = {
            han.input_x: x_,
            han.input_y: y_,
            # han.max_sentence_num: 1,
            han.max_sentence_length: FLAGS.max_word_in_sent,
            # han.batch_size: 1
        }
        batch_len = len(x_)
        loss, acc = sess.run([han.loss,han.acc],feed_dict = feed_dict )
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / val_len, total_acc / val_len



def train(x,y):
    with tf.Session() as sess:

        grad_summaries = []
        for g, v in han.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                grad_summaries.append(grad_hist_summary)

        grad_summaries_merged = tf.summary.merge(grad_summaries)

        loss_summary = tf.summary.scalar('loss', han.loss)
        acc_summary = tf.summary.scalar('accuracy', han.acc)

        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(han.out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(han.out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(han.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model" + pathStr)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        sess.run(tf.global_variables_initializer())

        best_accurate = 0.0
        last_update = 0
        total_batch = 0

        flag =False
        for epoch in range(FLAGS.num_epochs):
            for x_batch, y_batch in iter_batch(x, y, FLAGS.batch_size):
                feed_dict = {
                    han.input_x: x_batch,
                    han.input_y: y_batch,
                    # han.max_sentence_num: 1,
                    han.max_sentence_length: FLAGS.max_word_in_sent,
                    # han.batch_size: 1
                }
                _,step, summaries, cost, accuracy = sess.run([han.train_op,han.global_steps, dev_summary_op, han.loss, han.acc], feed_dict)
                # summaries, cost, accuracy = sess.run([dev_summary_op, han.loss, han.acc], feed_dict)
                print("step is ",step,"total is ",total_batch, " cost ", cost, " accuracy", accuracy)
                if step % 10 == 0:
                    train_summary_writer.add_summary(summaries, step)
                if step % 100 == 0:
                    # train_cost, train_accuracy = sess.run([han.loss, han.acc],feed_dict)
                    val_cost, val_accuracy = evaluate(sess, val_x, val_y)

                    if val_accuracy > best_accurate:
                        best_accurate = val_accuracy
                        last_update = step
                        saver.save(sess=sess, save_path=checkpoint_prefix)
                        restore = "update------"
                    else:
                        restore = "no_update"
                    time_str = str(int(time.time()))
                    msg ='Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                       + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    print(msg.format(step, cost, accuracy, val_cost, val_accuracy, time_str, restore))
                total_batch += 1

                if step - last_update > 2000:
                    print("the model no update for 2000 iter step, stop batch_iter at {} step".format(step))
                    flag = True
                    break
            if flag:
                print("stop the epoch loop at {}".format(epoch))
                break
        print("train done, because the valid accurate no update or the epoch is done")


def test():
    print("Loading test data...")
    x_test, y_test = test_x, test_y
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    modeldir = os.path.join('./rum/checkpoints/model'+pathStr)
    saver.restore(session,modeldir)

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)

    test_cost, test_accurate = evaluate(session, x_test, y_test)

    print("the test data cost is {} and accurate is {}".format(test_cost, test_accurate))

    for i in range(num_batch):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, data_len)
        feed_dict = {
            han.input_x: x_test[start_index:end_index],

            # han.max_sentence_num: 1,
            han.max_sentence_length: FLAGS.max_word_in_sent,
            # han.batch_size: 1
        }

        y_pred_cls[start_index:end_index] = session.run(han.predict, feed_dict=feed_dict)
    print(len(y_test_cls), len(y_pred_cls))
    print("Precision, Recall, F1-Score")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=list(label2id.keys())))

    print("Confusion Matrix")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)
if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")
    han = HAN(vocab_size=FLAGS.vocab_size,
                    num_classes=FLAGS.num_classes,
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size)

    pathStr = "first"
    train_x, train_y, test_x, test_y, val_x, val_y, label2id = load_dataset(FLAGS.yelp_json_path,
                                                                            FLAGS.max_word_in_sent)
    if sys.argv[1] == 'train':
        train(train_x,train_y)
    else:
        test()
    # train(train_x, train_y)



