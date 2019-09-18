import tensorflow as tf
import numpy as np
from instance_normalization import InstanceNormalization

FLAGS = tf.compat.v1.app.flags.FLAGS

class KerasModel(tf.keras.Model):
    """ Keras Model common functions class
    Define common functions for discriminator and generator
    # Arguments
        name (optional): name of the class
    """
    def __init__(self, name=""):
        super(KerasModel, self).__init__(name=name)

    def conv_block(self, num_filters, kernel_size=3):
        conv_1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")
        conv_2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", padding="same")
        return conv_1, conv_2

    def norm(self, input_tensor, batch_norm=False, instance_norm=False):
        if batch_norm:
            input_tensor = self.bn(input_tensor)
        if instance_norm:
            input_tensor = self.instance_normalization(input_tensor)
        return input_tensor

class Generator(KerasModel):
    """ Generator Model
    Define generator class
    # Arguments
        num_filters_base: number of base filters
        name (optional): name of the class
    """
    def __init__(self, num_filters_base=16, name="generator"):
        super(Generator, self).__init__(name=name)
        self.num_filters_base = num_filters_base
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.instance_normalization = InstanceNormalization(axis=-1)
        self.p = tf.keras.layers.MaxPooling2D(pool_size=[2, 2])
        self.c1_1, self.c1_2 = self.conv_block(num_filters=self.num_filters_base * 1)
        self.c2_1, self.c2_2 = self.conv_block(num_filters=self.num_filters_base * 2)
        self.c3_1, self.c3_2 = self.conv_block(num_filters=self.num_filters_base * 4)
        self.c4_1, self.c4_2 = self.conv_block(num_filters=self.num_filters_base * 8)
        self.c5_1, self.c5_2 = self.conv_block(num_filters=self.num_filters_base * 16)
        self.u6 = tf.keras.layers.Conv2DTranspose(filters=self.num_filters_base * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.c6_1, self.c6_2 = self.conv_block(num_filters=self.num_filters_base * 8)
        self.u7 = tf.keras.layers.Conv2DTranspose(filters=self.num_filters_base * 4, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.c7_1, self.c7_2 = self.conv_block(num_filters=self.num_filters_base * 4)
        self.u8 = tf.keras.layers.Conv2DTranspose(filters=self.num_filters_base * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.c8_1, self.c8_2 = self.conv_block(num_filters=self.num_filters_base * 2)
        self.u9 = tf.keras.layers.Conv2DTranspose(filters=self.num_filters_base * 1, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.c9_1, self.c9_2 = self.conv_block(num_filters=self.num_filters_base * 1)
        self.c10 = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), activation="sigmoid", padding="same")

    def __call__(self, input_tensor, dropout_rate, batch_norm=False, instance_norm=False):
        self.d = tf.keras.layers.Dropout(rate=dropout_rate)
        """ Contraction Layer """
        """ Layer 1 """
        x = self.c1_1(input_tensor)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c1_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        c1 = x
        x = self.p(x)
        x = self.d(x)
        """ Layer 2 """
        x = self.c2_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c2_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        c2 = x
        x = self.p(x)
        x = self.d(x)
        """ Layer 3 """
        x = self.c3_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c3_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        c3 = x
        x = self.p(x)
        x = self.d(x)
        """ Layer 4 """
        x = self.c4_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c4_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        c4 = x
        x = self.p(x)
        x = self.d(x)
        """ Layer 5 """
        x = self.c5_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c5_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        """ Expansive Layer """
        """ Layer 6 """
        x = self.u6(x)
        x = tf.keras.layers.concatenate([x, c4])
        x = self.d(x)
        x = self.c6_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c6_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        """ Layer 7 """
        x = self.u7(x)
        x = tf.keras.layers.concatenate([x, c3])
        x = self.d(x)
        x = self.c7_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c7_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        """ Layer 8 """
        x = self.u8(x)
        x = tf.keras.layers.concatenate([x, c2])
        x = self.d(x)
        x = self.c8_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c8_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        """ Layer 9 """
        x = self.u9(x)
        x = tf.keras.layers.concatenate([x, c1])
        x = self.d(x)
        x = self.c9_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c9_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        """ Layer 10 """
        x = self.c10(x)
        return x

class Discriminator(KerasModel):
    """ Discriminator Model
    Define discriminator class
    # Arguments
        num_filters_base: number of base filters
        name (optional): name of the class
    """
    def __init__(self, num_filters_base=16, name="discriminator"):
        super(Discriminator, self).__init__(name=name)
        self.num_filters_base = num_filters_base
        self.bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.instance_normalization = InstanceNormalization(axis=-1)
        self.p1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2])
        self.p2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[1, 1])
        self.c1_1, self.c1_2 = self.conv_block(num_filters=self.num_filters_base * 4)
        self.c2_1, self.c2_2 = self.conv_block(num_filters=self.num_filters_base * 8)
        self.c3_1, self.c3_2 = self.conv_block(num_filters=self.num_filters_base * 16)
        self.c4_1, self.c4_2 = self.conv_block(num_filters=self.num_filters_base * 32)
        self.c5 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same")
        self.activation = tf.keras.layers.Activation("sigmoid")

    def __call__(self, real_tensor, fake_tensor, dropout_rate, batch_norm=False, instance_norm=False):
        self.d = tf.keras.layers.Dropout(rate=dropout_rate)
        input_tensor = tf.keras.layers.concatenate([real_tensor, fake_tensor])
        """ Layer 1 """
        x = self.c1_1(input_tensor)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c1_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.p1(x)
        x = self.d(x)
        """ Layer 2 """
        x = self.c2_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c2_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.p1(x)
        x = self.d(x)
        """ Layer 3 """
        x = self.c3_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c3_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.p1(x)
        x = self.d(x)
        """ Layer 4 """
        x = self.c4_1(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.c4_2(x)
        x = self.norm(input_tensor=x, batch_norm=batch_norm, instance_norm=instance_norm)
        x = self.p2(x)
        x = self.d(x)
        """ Layer 5 """
        x = self.c5(x)
        x = self.p2(x)
        x = self.d(x)
        
        x = self.activation(x)
        return x

class ImageGrainer(object):
    """ ImageGrainer Model
    Define image grainer class
    # Arguments
        num_filters_base: number of base filters
        batch_norm: Whether to use batch normalization or not 
        dropout: dropout rate to be applied
        instance_norm: Whether to use instance normalization or not
        discriminator_lambda: the weight of discriminator loss
        generator_lambda: the weight of generator loss
        cycle_lambda: weight of cycle loss
        identity_lambda: weight of identity loss
    """
    def __init__(self, num_filters_base=16, batch_norm=False, dropout=0.1, instance_norm=False, discriminator_lambda=0.5, generator_lambda=1, cycle_lambda=10, identity_lambda=5):
        super(ImageGrainer, self).__init__()
        self.real_x = tf.keras.backend.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        self.real_y = tf.keras.backend.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
        self.dropout = tf.keras.backend.placeholder(dtype=tf.float32)
        self.num_filters_base = num_filters_base
        self.batch_norm = batch_norm
        self.instance_norm = instance_norm
        self.loss_obj = tf.keras.losses.BinaryCrossentropy()
        self.discriminator_lambda = discriminator_lambda
        self.generator_lambda = generator_lambda
        self.cycle_lambda = cycle_lambda
        self.identity_lambda = identity_lambda

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_loss = real_loss + generated_loss
        return total_loss * self.discriminator_lambda

    def generator_loss(self, generated):
        total_loss = self.loss_obj(tf.ones_like(generated), generated)
        return total_loss * self.generator_lambda

    def calc_cycle_loss(self, real_image, cycled_image):
        cycle_loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return cycle_loss * self.cycle_lambda

    def calc_identity_loss(self, real_image, same_image):
        identity_loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return identity_loss * self.identity_lambda


    def create_model(self):
        # Generator G translates X -> Y
        # Generator F translates Y -> X.
        generator_g = Generator(num_filters_base=self.num_filters_base, name="generator_g")
        generator_f = Generator(num_filters_base=self.num_filters_base, name="generator_f")
        discriminator_x = Discriminator(num_filters_base=self.num_filters_base, name="discriminator_x")
        discriminator_y = Discriminator(num_filters_base=self.num_filters_base, name="discriminator_y")

        args = [self.dropout, self.batch_norm, self.instance_norm]

        self.fake_y = generator_g(self.real_x, *args)
        self.cycled_x = generator_f(self.fake_y, *args)

        self.fake_x = generator_f(self.real_y, *args)
        self.cycled_y = generator_g(self.fake_x, *args)

        self.same_x = generator_f(self.real_x, *args)
        self.same_y = generator_g(self.real_y, *args)

        self.disc_real_x = discriminator_x(self.real_y, self.real_x, *args)
        self.disc_real_y = discriminator_y(self.real_x, self.real_y, *args)

        self.disc_fake_x = discriminator_x(self.real_y, self.fake_x, *args)
        self.disc_fake_y = discriminator_y(self.real_x, self.fake_y, *args)

        self.gen_g_loss = self.generator_loss(self.disc_fake_y)
        self.gen_f_loss = self.generator_loss(self.disc_fake_x)

        self.total_cycle_loss = self.calc_cycle_loss(self.real_x, self.cycled_x) + self.calc_cycle_loss(self.real_y, self.cycled_y)

        self.total_gen_g_loss = self.gen_g_loss + self.total_cycle_loss + self.calc_identity_loss(self.real_y, self.same_y)
        self.total_gen_f_loss = self.gen_f_loss + self.total_cycle_loss + self.calc_identity_loss(self.real_x, self.same_x)

        self.disc_x_loss = self.discriminator_loss(self.disc_real_x, self.disc_fake_x)
        self.disc_y_loss = self.discriminator_loss(self.disc_real_y, self.disc_fake_y)

        self.generator_g_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0.5)
        self.generator_f_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0.5)

        self.discriminator_x_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0.5)
        self.discriminator_y_optimizer = tf.compat.v1.train.AdamOptimizer(1e-4, beta1=0.5)

        self.generator_g_optimizer = self.generator_g_optimizer.minimize(self.total_gen_g_loss, var_list=generator_g.trainable_variables)
        self.generator_f_optimizer = self.generator_f_optimizer.minimize(self.total_gen_f_loss, var_list=generator_f.trainable_variables)

        self.discriminator_x_optimizer = self.discriminator_x_optimizer.minimize(self.disc_x_loss, var_list=discriminator_x.trainable_variables)
        self.discriminator_y_optimizer = self.discriminator_y_optimizer.minimize(self.disc_y_loss, var_list=discriminator_y.trainable_variables)


def main(_):
    image_grainer = ImageGrainer(FLAGS.num_filters_base, FLAGS.batch_norm, FLAGS.dropout, FLAGS.instance_norm)
    image_grainer.create_model()
    print(image_grainer.discriminator_x_optimizer)

if __name__ == "__main__":
    tf.compat.v1.app.flags.DEFINE_integer("num_filters_base", 1, "number of filters")
    tf.compat.v1.app.flags.DEFINE_boolean("batch_norm", False, "batch norm")
    tf.compat.v1.app.flags.DEFINE_boolean("instance_norm", False, "batch norm")
    tf.compat.v1.app.flags.DEFINE_float("dropout", 0.1, "dropout")
    tf.compat.v1.app.run()
