import tensorflow as tf


def classifier_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)

    return loss


def dcgan_loss():
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def d_loss(real_logits, fake_logits):
        real_loss = criterion(tf.ones_like(real_logits), real_logits)
        fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)
        return real_loss + fake_loss

    def g_loss(fake_logits):
        return criterion(tf.ones_like(fake_logits), fake_logits)

    return d_loss, g_loss


def d_loss(real_logits, fake_logits):
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = criterion(tf.ones_like(real_logits), real_logits)
    fake_loss = criterion(tf.zeros_like(fake_logits), fake_logits)

    return real_loss + fake_loss


def g_loss(fake_logits):
    criterion = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    return criterion(tf.ones_like(fake_logits), fake_logits)


def wgan_loss():

    def d_loss(real_logits, fake_logits):
        return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

    def g_loss(fake_logits):
        return -tf.reduce_mean(fake_logits)

    return d_loss, g_loss