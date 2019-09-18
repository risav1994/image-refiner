import tensorflow as tf
import h5py
import numpy as np
import cv2
from image_degrade_model import ImageGrainer
FLAGS = tf.compat.v1.app.flags.FLAGS

def main(_):
    train_x = np.load("output_data_" + FLAGS.train_h5)
    train_y = np.load("input_data_" + FLAGS.train_h5)
    
    batch_size = FLAGS.batch_size
    total_batches = int(len(train_x) / batch_size)
    if len(train_x) % batch_size != 0:
        total_batches += 1

    graph = tf.Graph()
    previous_loss = 1e10
    previous_best = (-1, -1)
    with graph.as_default():
        sess = tf.Session()
        model = ImageGrainer(FLAGS.num_filters_base, FLAGS.batch_norm, FLAGS.dropout, FLAGS.instance_norm)
        model.create_model()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(FLAGS.ckpt_dir + "model-2-197.meta")
        saver.restore(sess, FLAGS.ckpt_dir + "model-2-197")
        
        counter = 0
        for batch in range(total_batches):
            counter += 1
            batch_start = int(batch * batch_size)
            batch_end = int((batch + 1) * batch_size)
            if batch == total_batches - 1:
                feed_dict = {model.real_x: train_x[batch_start: ], model.real_y: train_y[batch_start: ], model.dropout: 0.0}
            else:
                feed_dict = {model.real_x: train_x[batch_start: batch_end], model.real_y: train_y[batch_start: batch_end], model.dropout: 0.0}
            image_out, image_in = sess.run([model.fake_y, model.real_x], feed_dict=feed_dict)
            for idx, image in enumerate(image_out):
                image_refined = image * 255
                image_refined = image_refined.astype(np.uint8)
                image_in_ = image_in[idx] * 255
                image_in_ = image_in_.astype(np.uint8)
                images = np.concatenate((image_in_, image_refined), axis=1)
                cv2.imwrite("results/img_" + repr(batch * batch_size + idx) + ".png", images)




if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_integer("num_filters_base", 1, "number of filters")
    tf.compat.v1.app.flags.DEFINE_boolean("batch_norm", False, "batch norm")
    tf.compat.v1.app.flags.DEFINE_boolean("instance_norm", False, "batch norm")
    tf.compat.v1.app.flags.DEFINE_float("dropout", 0.1, "dropout")
    tf.compat.v1.app.flags.DEFINE_string("ckpt_dir", "checkpoint_dir", "checkpoint directory")
    tf.compat.v1.app.flags.DEFINE_string("train_h5", "train.h5", "train h5py file")
    tf.compat.v1.app.flags.DEFINE_string("test_h5", "test.h5", "test h5py file")
    tf.compat.v1.app.flags.DEFINE_integer("batch_size", 32, "batch size")
    tf.compat.v1.app.flags.DEFINE_integer("num_epochs", 1000, "batch size")
    tf.compat.v1.app.run()