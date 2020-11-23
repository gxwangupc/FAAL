import tensorflow as tf
import numpy as np
import random
from scipy.spatial import distance_matrix
from config import Config

opt = Config().parse()


class al_acquisition:
    def __init__(self):
        super(al_acquisition, self).__init__()

    def sample(self, classifier, discriminator, data_unl, budget):
        all_scores = []
        all_indices = []

        for idx, img in enumerate(data_unl):
            img = np.expand_dims(img, axis=0)
            fea, _ = classifier(img, training=False)
            score = discriminator(fea, training=False)

            all_scores.extend(score)
            all_indices.append(idx)

        all_scores = tf.stack(all_scores)
        all_scores = np.array(all_scores).reshape(-1)
        '''need to multiply by -1 to be able to use tf.math.top_k'''
        all_scores *= -1

        '''querry the top K minimum'''
        _, querry_indices = tf.math.top_k(all_scores, int(budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices






