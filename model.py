from tensorflow import keras
from config import Config

opt = Config().parse()


"""
Classifier.
"""
class classifier(keras.Model):
    def __init__(self):
        super(classifier, self).__init__()

        self.fe = keras.Sequential()
        self.fe.add(keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu', name='cnn_conv3d_1'))
        self.fe.add(keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu', name='cnn_conv3d_2'))
        self.fe.add(keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', name='cnn_conv3d_3'))
        if opt.CHANNEL == 15:
            self.fe.add(keras.layers.Reshape((19, 19, 96), name='cnn_reshape'))
        elif opt.CHANNEL == 20:
            self.fe.add(keras.layers.Reshape((19, 19, 256), name='cnn_reshape'))
        elif opt.CHANNEL == 25:
            self.fe.add(keras.layers.Reshape((19, 19, 416), name='cnn_reshape'))
        elif opt.CHANNEL == 30:
            self.fe.add(keras.layers.Reshape((19, 19, 576), name='cnn_reshape'))
        self.fe.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cnn_conv2d_1'))
        self.fe.add(keras.layers.Flatten(name='cnn_flatten'))

        self.mlp = keras.Sequential()
        self.mlp.add(keras.layers.Dense(units=256, activation='relu', name='cnn_dense_1'))
        self.mlp.add(keras.layers.Dropout(opt.DR_RATE, name='cnn_dropout_1'))
        self.mlp.add(keras.layers.Dense(units=128, activation='relu', name='cnn_dense_2'))
        self.mlp.add(keras.layers.Dropout(opt.DR_RATE, name='cnn_dropout_2'))
        self.mlp.add(keras.layers.Dense(units=opt.N_CLS, activation='softmax', name='cnn_output'))

    def call(self, input, training=None):
        fea = self.fe(input, training=training)
        output = self.mlp(fea, training=training)
        return fea, output


"""
Convolutional part of classifier.
"""
class fe(keras.Model):
    def __init__(self):
        super(fe, self).__init__()

        self.fe = keras.Sequential()
        self.fe.add(keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu', name='cnn_conv3d_1'))
        self.fe.add(keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu', name='cnn_conv3d_2'))
        self.fe.add(keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', name='cnn_conv3d_3'))
        if opt.CHANNEL == 15:
            self.fe.add(keras.layers.Reshape((19, 19, 96), name='cnn_reshape'))
        elif opt.CHANNEL == 20:
            self.fe.add(keras.layers.Reshape((19, 19, 256), name='cnn_reshape'))
        elif opt.CHANNEL == 25:
            self.fe.add(keras.layers.Reshape((19, 19, 416), name='cnn_reshape'))
        elif opt.CHANNEL == 30:
            self.fe.add(keras.layers.Reshape((19, 19, 576), name='cnn_reshape'))
        self.fe.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cnn_conv2d_1'))
        self.fe.add(keras.layers.Flatten(name='cnn_flatten'))

    def call(self, input, training=None):
        fea = self.fe(input, training=training)
        return fea


"""
Fully connected part of classifier.
"""
class mlp(keras.Model):
    def __init__(self):
        super(mlp, self).__init__()

        self.mlp = keras.Sequential()
        self.mlp.add(keras.layers.Dense(units=256, activation='relu', name='cnn_dense_1'))
        self.mlp.add(keras.layers.Dropout(opt.DR_RATE, name='cnn_dropout_1'))
        self.mlp.add(keras.layers.Dense(units=128, activation='relu', name='cnn_dense_2'))
        self.mlp.add(keras.layers.Dropout(opt.DR_RATE, name='cnn_dropout_2'))
        self.mlp.add(keras.layers.Dense(units=opt.N_CLS, activation='softmax', name='cnn_output'))

    def call(self, input, training=None):
        output = self.mlp(input, training=training)
        return output


"""
Feature generator.
"""
class generator(keras.Model):
    def __init__(self):
        super(generator, self).__init__()

        self.g_dense_1 = keras.layers.Dense(units=4*4*512, activation='relu')
        self.g_bn_dense_1 = keras.layers.BatchNormalization()
        self.g_reshape = keras.layers.Reshape((4, 4, 512))
        self.g_conv2dt_1 = keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same', strides=(2, 2),
                                                        activation='relu')
        self.g_bn_conv_1 = keras.layers.BatchNormalization()
        self.g_conv2dt_2 = keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', strides=(2, 2),
                                                        activation='relu')
        self.g_bn_conv_2 = keras.layers.BatchNormalization()
        self.g_conv2dt_3 = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='valid', strides=(1, 1),
                                                        activation='relu')
        self.g_flatten = keras.layers.Flatten()

    def call(self, input, training=None):
        # out_dense_1 = self.g_dense_1(input, training=training)
        # out_dense_1 = self.g_reshape(out_dense_1)
        # out_conv_1 = self.g_conv2dt_1(out_dense_1, training=training)
        # out_conv_2 = self.g_conv2dt_2(out_conv_1, training=training)
        out_dense_1 = self.g_bn_dense_1(self.g_dense_1(input, training=training))
        out_dense_1 = self.g_reshape(out_dense_1)
        out_conv_1 = self.g_bn_conv_1(self.g_conv2dt_1(out_dense_1, training=training))
        out_conv_2 = self.g_bn_conv_2(self.g_conv2dt_2(out_conv_1, training=training))
        out_conv_2 = out_conv_2[:,:15,:15,:]
        out_conv_3 = self.g_conv2dt_3(out_conv_2, training=training)
        output = self.g_flatten(out_conv_3, training=training)
        return output


class generator_mlp(keras.Model):
    def __init__(self):
        super(generator_mlp, self).__init__()

        self.g_mlp = keras.Sequential()
        self.g_mlp.add(keras.layers.Dense(units=256, activation='relu'))
        self.g_mlp.add(keras.layers.Dropout(opt.DR_RATE))
        self.g_mlp.add(keras.layers.Dense(units=512, activation='relu'))
        self.g_mlp.add(keras.layers.Dropout(opt.DR_RATE))
        self.g_mlp.add(keras.layers.Dense(units=17*17*64))
        self.g_flatten = keras.layers.Flatten()

    def call(self, input, training=None):
        output = self.g_flatten(self.g_mlp(input, training=training))
        return output

"""
Feature discriminator.
"""
class discriminator(keras.Model):
    def __init__(self):
        super(discriminator, self).__init__()

        self.mlp = keras.Sequential()
        self.mlp.add(keras.layers.Dense(units=512, name='d_dense_1'))
        self.mlp.add(keras.layers.LeakyReLU(alpha=0.2, name='d_lrelu_1'))
        self.mlp.add(keras.layers.Dropout(opt.DR_RATE, name='d_dropout_1'))
        self.mlp.add(keras.layers.Dense(units=512, name='d_dense_2'))
        self.mlp.add(keras.layers.LeakyReLU(alpha=0.2, name='d_lrelu_2'))
        self.mlp.add(keras.layers.Dropout(opt.DR_RATE, name='d_dropout_2'))
        self.mlp.add(keras.layers.Dense(units=1, activation='sigmoid', name='output'))

    def call(self, input, training=None):
        output = self.mlp(input, training=training)
        return output




