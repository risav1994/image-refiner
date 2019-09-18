import tensorflow as tf
import h5py
import numpy as np
import random
from image_degrade_model import ImageGrainer
FLAGS = tf.compat.v1.app.flags.FLAGS

def main(_):
    train_x = np.load("output_data_" + FLAGS.train_h5)
    train_y = np.load("input_data_" + FLAGS.train_h5)
    combined_train = list(zip(train_x, train_y))
    random.shuffle(combined_train)
    train_x, train_y = zip(*combined_train)
    test_x = np.load("output_data_" + FLAGS.test_h5)
    test_y = np.load("input_data_" + FLAGS.test_h5)
    combined_test = list(zip(test_x, test_y))
    random.shuffle(combined_test)
    test_x, test_y = zip(*combined_test)
    batch_size = FLAGS.batch_size
    total_batches = int(len(train_x) / batch_size)
    if len(train_x) % batch_size != 0:
        total_batches += 1
    total_test_batches = int(len(test_x) / batch_size)
    if len(test_x) % batch_size != 0:
        total_test_batches += 1
    graph = tf.Graph()
    previous_loss = 1e10
    previous_best = (-1, -1)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with graph.as_default():
        sess = tf.Session(config=config)
        model = ImageGrainer(FLAGS.num_filters_base, FLAGS.batch_norm, FLAGS.dropout, FLAGS.instance_norm)
        model.create_model()
        sess.run(tf.global_variables_initializer())
        counter = 0
        for epoch in range(FLAGS.num_epochs):
            for batch in range(total_batches):
                counter += 1
                batch_start = int(batch * batch_size)
                batch_end = int((batch + 1) * batch_size)
                if batch == total_batches - 1:
                    feed_dict = {model.real_x: train_x[batch_start: ], model.real_y: train_y[batch_start: ], model.dropout: FLAGS.dropout}
                else:
                    feed_dict = {model.real_x: train_x[batch_start: batch_end], model.real_y: train_y[batch_start: batch_end], model.dropout: FLAGS.dropout}
                gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss, _, _, _, _ = sess.run([model.total_gen_g_loss, model.total_gen_f_loss, model.disc_x_loss, model.disc_y_loss, 
                    model.generator_g_optimizer, model.generator_f_optimizer, model.discriminator_x_optimizer, model.discriminator_y_optimizer], feed_dict=feed_dict)

                print("Epoch:", epoch, "Batch:", batch, "Generator Forward Loss:", gen_g_loss, "Generator Backward Loss:", gen_f_loss, "Discriminator Forward Loss:", disc_y_loss, "Discriminator Backward Loss:", disc_x_loss)
                if counter % 100 == 0:
                    gen_forward_loss = 0
                    gen_backward_loss = 0
                    disc_forward_loss = 0
                    disc_backward_loss = 0
                    for batch_test in range(total_test_batches):
                        batch_start = int(batch_test * batch_size)
                        batch_end = int((batch_test + 1) * batch_size)
                        if batch_test == total_test_batches - 1:
                            feed_dict_test = {model.real_x: test_x[batch_start: ], model.real_y: test_y[batch_start: ], model.dropout: 0.0}
                        else:
                            feed_dict_test = {model.real_x: test_x[batch_start: batch_end], model.real_y: test_y[batch_start: batch_end], model.dropout: 0.0}
                        gen_g_loss, gen_f_loss, disc_x_loss, disc_y_loss = sess.run([
                            model.total_gen_g_loss, model.total_gen_f_loss, model.disc_x_loss, model.disc_y_loss], feed_dict=feed_dict_test)
                        gen_forward_loss += gen_g_loss
                        gen_backward_loss += gen_f_loss
                        disc_forward_loss += disc_y_loss
                        disc_backward_loss += disc_x_loss
                    total_loss = (gen_forward_loss + gen_backward_loss + disc_forward_loss + disc_backward_loss) / total_test_batches
                    print("Test ======> Epoch:", epoch, "Batch:", batch, "Generator Forward Loss:", gen_forward_loss / total_test_batches, "Generator Backward Loss:", gen_backward_loss / total_test_batches, "Discriminator Forward Loss:", disc_forward_loss / total_test_batches, "Discriminator Backward Loss:", disc_backward_loss / total_test_batches, "Total Loss: ", total_loss)
                    print("Previous Best (Epoch, Batch):", (previous_best), "Previous Best Loss:", previous_loss)
                    if total_loss < previous_loss:
                        saver = tf.train.Saver()
                        saver.save(sess, FLAGS.ckpt_dir + "model-" + repr(epoch) + "-" + repr(batch))
                        previous_loss = total_loss
                        previous_best = (epoch, batch)


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
