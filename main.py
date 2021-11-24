import tensorflow as tf
from tensorflow import keras
from dataloader import load, apply_pca, pad_zeros, get_init_indices, get_data
from model import classifier, generator, discriminator
from losses import classifier_loss, dcgan_loss
from acquisition import al_acquisition

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import numpy as np
import os
import spectral
from config import Config

opt = Config().parse()

if not os.path.exists(opt.RESULT):
    os.makedirs(opt.RESULT)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPU)
print("using gpu {}".format(opt.GPU))

"""
Data preparation.
"""
opt.use_PCA = True
opt.use_SuperPCA = False
print(opt.use_PCA)
print(opt.use_SuperPCA)
assert (opt.use_PCA and opt.use_SuperPCA) == False
data, label = load(opt.DATASET)
if opt.use_PCA:
    data, pca = apply_pca(data)
else:
    pass

init_tr_labeled_idx, init_tr_unlabeled_idx, te_idx = get_init_indices(data, label)

init_trl_data, init_trl_label = get_data(data, label, init_tr_labeled_idx)
init_trunl_data, init_trunl_label = get_data(data, label, init_tr_unlabeled_idx)
te_data, te_label = get_data(data, label, te_idx)

init_trl_data = np.expand_dims(init_trl_data, axis=4)
init_trl_label = keras.utils.to_categorical(init_trl_label)
init_trunl_data = np.expand_dims(init_trunl_data, axis=4)
init_trunl_label = keras.utils.to_categorical(init_trunl_label)
te_data = np.expand_dims(te_data, axis=4)
te_label = keras.utils.to_categorical(te_label)

init_trl_set = tf.data.Dataset.from_tensor_slices((init_trl_data, init_trl_label)).shuffle(len(init_trl_data)).batch(opt.BATCH_SIZE)
te_set = tf.data.Dataset.from_tensor_slices((te_data, te_label)).batch(opt.BATCH_SIZE)


"""
Model.
"""
'''create classifier'''
classifier = classifier()
optim = keras.optimizers.Adam(lr=opt.LR, decay=opt.DECAY)
classifier.build(input_shape=(opt.BATCH_SIZE, opt.WINDOW_SIZE, opt.WINDOW_SIZE, opt.CHANNEL, 1))
classifier.summary()

'''create feature generator & feature discriminator'''
fea_g = generator()
fea_g.build(input_shape=(opt.BATCH_SIZE, opt.DIM_Z))
fea_g.summary()

fea_d = discriminator()
fea_d.build(input_shape=(opt.BATCH_SIZE, 17*17*64))
fea_d.summary()

d_loss, g_loss = dcgan_loss()
fea_g_optim = keras.optimizers.Adam(learning_rate=opt.GAN_LR, beta_1=0.5)
fea_d_optim = keras.optimizers.Adam(learning_rate=opt.GAN_LR, beta_1=0.5)

"""
Test Function.
"""

def test_func(te_data, te_label):
    print("Start testing using test data!")
    _, te_pred = classifier.predict(te_data)
    te_label = tf.argmax(te_label, axis=1)
    te_pred = tf.argmax(te_pred, axis=1)

    classification = classification_report(te_label, te_pred)
    print(classification)
    classification, confusion, oa, each_acc, aa, kappa = reports(te_data, te_label)
    classification = str(classification)
    confusion = str(confusion)
    report = os.path.join(opt.RESULT,
                          str(opt.DATASET) + "_" + str(opt.ACQUISITION) + "_loop" + str(loop) + "_report.txt")

    with open(report, 'w') as f:
        f.write('\n')
        f.write("DATASET:{}".format(str(opt.DATASET)))
        f.write('\n')
        f.write("WINDOW_SIZE:{}, use_SuperPCA:{}, CHANNEL:{}, BATCH_SIZE:{}, BUDGET:{}".format(str(opt.WINDOW_SIZE),
                                                                                               str(opt.use_SuperPCA),
                                                                                               str(opt.CHANNEL),
                                                                                               str(opt.BATCH_SIZE),
                                                                                               str(opt.BUDGET)))
        f.write('\n')
        f.write('\n')
        f.write('\n')
        f.write('Each: {}'.format(each_acc))
        f.write('\n')
        f.write('{} Kappa accuracy (%)'.format(kappa))
        f.write('\n')
        f.write('{} Overall accuracy (%)'.format(oa))
        f.write('\n')
        f.write('{} Average accuracy (%)'.format(aa))
        f.write('\n')
        f.write('\n')
        f.write('{}'.format(classification))
        f.write('\n')
        f.write('{}'.format(confusion))


def aa_and_each_acc(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports(data, label):
    _, _te_pred = classifier.predict(data)
    _te_pred = np.argmax(_te_pred, axis=1)
    if opt.DATASET == 'Indian':
        targets = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                   'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                   'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                   'Stone-Steel-Towers']
    elif opt.DATASET == 'Salinas':
        targets = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                   'Fallow_smooth',
                   'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                   'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                   'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif opt.DATASET == 'PaviaU':
        targets = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                   'Self-Blocking Bricks', 'Shadows']
    elif opt.DATASET == 'KSC':
        targets = ['Scrub', 'Willow Swamp', 'Cabbage Palm Hammock', 'Cabbage Palm/Oak', 'Slash Pine',
                   'Oak/Broadleaf Hammock', 'Hardwood Swamp', 'Graminoid Marsh', 'Spartina Marsh',
                   'Cattail Marsh', 'Salt Marsh', 'Mud Flats', 'Water']
    elif opt.DATASET == 'Pavia':
        targets = ['Water', 'Trees', 'Asphalt', 'Self-Blocking-Bricks', 'Bitumen', 'Tiles', 'Shadows',
                   'Meadows', 'Bare soil']
    else:
        raise NotImplementedError

    classification = classification_report(label, _te_pred, target_names=targets)
    oa = accuracy_score(label, _te_pred)
    confusion = confusion_matrix(label, _te_pred)
    each_acc, aa = aa_and_each_acc(confusion)
    kappa = cohen_kappa_score(label, _te_pred)

    return classification, confusion, oa * 100, each_acc * 100, aa * 100, kappa * 100


"""
Pre-train classifier with original labeled data.
"""
print("Start training classifier using initial labeled training data!")
init_te_metric = tf.keras.metrics.Accuracy()
for epoch in range(opt.INIT_EPOCH):

    for step, (_data, _label) in enumerate(init_trl_set):

        with tf.GradientTape() as tape:
            _fea, _prediction = classifier(_data, training=True)
            _loss = classifier_loss(_label, _prediction)

        grads = tape.gradient(_loss, classifier.trainable_variables)
        optim.apply_gradients(zip(grads, classifier.trainable_variables))


    # init_te_metric.reset_states()
    # for x, y in te_set:
    #     _, _te_pred = classifier(x, training=False)
    #     y = tf.argmax(y, axis=1)
    #     _te_pred = tf.argmax(_te_pred, axis=1)
    #     init_te_metric.update_state(y, _te_pred)
    #
    # print("initial testing:", "epoch", epoch, "acc:", init_te_metric.result().numpy())

loop = 0
if opt.ACQUISITION == 'al_acquisition':
    """
    Freeze the training of classifier and obtain the intermediate features of initial labeled data.
    """
    init_trl_fea = []
    for step, (initl_data, initl_label) in enumerate(init_trl_set):
        initl_fea, _ = classifier(initl_data, training=False)
        init_trl_fea.extend(initl_fea)

    init_trl_fea = np.array(init_trl_fea)
    init_trl_fea_set = tf.data.Dataset.from_tensor_slices(init_trl_fea).shuffle(len(init_trl_fea)).batch(opt.BATCH_SIZE)


    """
    Train GAN using intermediate features of initial labeled training data.
    """
    '''training step for feature generator & feature discriminator'''
    @tf.function
    def train_d_step(fea):
        noise = tf.random.normal(shape=[opt.BATCH_SIZE, opt.DIM_Z])
        with tf.GradientTape() as disc_tape:
            gen_fea = fea_g(noise, training=True)
            disc_real = fea_d(fea, training=True)
            disc_fake = fea_d(gen_fea, training=True)
            disc_loss = tf.reduce_mean(d_loss(disc_real, disc_fake))
        grads_d = disc_tape.gradient(disc_loss, fea_d.trainable_variables)
        fea_d_optim.apply_gradients(zip(grads_d, fea_d.trainable_variables))

        return disc_loss

    @tf.function
    def train_g_step():
        noise = tf.random.normal(shape=[opt.BATCH_SIZE, opt.DIM_Z])
        with tf.GradientTape() as gen_tape:
            gen_fea = fea_g(noise, training=True)
            disc_fake = fea_d(gen_fea, training=True)
            gen_loss = tf.reduce_mean(g_loss(disc_fake))
        grads_g = gen_tape.gradient(gen_loss, fea_g.trainable_variables)
        fea_g_optim.apply_gradients(zip(grads_g, fea_g.trainable_variables))

        return gen_loss

    @tf.function
    def train_d_msgan_step(fea):
        noise_1 = tf.random.normal(shape=[opt.BATCH_SIZE, opt.DIM_Z])
        noise_2 = tf.random.normal(shape=[opt.BATCH_SIZE, opt.DIM_Z])
        with tf.GradientTape() as disc_tape:
            gen_fea_1 = fea_g(noise_1, training=True)
            gen_fea_2 = fea_g(noise_2, training=True)
            disc_real = fea_d(fea, training=True)
            disc_fake_1 = fea_d(gen_fea_1, training=True)
            disc_fake_2 = fea_d(gen_fea_2, training=True)
            disc_loss = tf.reduce_mean(d_loss(disc_real, disc_fake_1)) + tf.reduce_mean(d_loss(disc_real, disc_fake_2))
        grads_d = disc_tape.gradient(disc_loss, fea_d.trainable_variables)
        fea_d_optim.apply_gradients(zip(grads_d, fea_d.trainable_variables))

        return disc_loss

    @tf.function
    def train_g_msgan_step():
        noise_1 = tf.random.normal(shape=[opt.BATCH_SIZE, opt.DIM_Z])
        noise_2 = tf.random.normal(shape=[opt.BATCH_SIZE, opt.DIM_Z])
        with tf.GradientTape() as gen_tape:
            gen_fea_1 = fea_g(noise_1, training=True)
            gen_fea_2 = fea_g(noise_2, training=True)
            disc_fake_1 = fea_d(gen_fea_1, training=True)
            disc_fake_2 = fea_d(gen_fea_2, training=True)
            ms_loss = tf.reduce_mean(tf.math.abs(gen_fea_1 - gen_fea_2)) / tf.reduce_mean(tf.math.abs(noise_1 - noise_2))
            eps = 1 * 1e-5
            ms_loss = 1 / (ms_loss + eps)
            gen_loss = tf.reduce_mean(g_loss(disc_fake_1)) + tf.reduce_mean(g_loss(disc_fake_2)) + ms_loss
        grads_g = gen_tape.gradient(gen_loss, fea_g.trainable_variables)
        fea_g_optim.apply_gradients(zip(grads_g, fea_g.trainable_variables))

        return gen_loss


    print("Start training GAN using the feature of initial labeled training data!")
    for epoch in range(opt.INIT_GAN_EPOCH):
        for step, _fea in enumerate(init_trl_fea_set):
            if opt.use_MS:
                disc_loss = train_d_msgan_step(_fea)
                gen_loss = train_g_msgan_step()
            else:
                disc_loss = train_d_step(_fea)
                gen_loss = train_g_step()


"""
Select unlabeled data and start active training.
"""
if opt.ACQUISITION == 'al_acquisition':
    al_acquisition_func = al_acquisition()

current_trl_idx = init_tr_labeled_idx
current_trunl_idx = init_tr_unlabeled_idx
current_trl_data = init_trl_data
current_trunl_data = init_trunl_data


al_te_metric = tf.keras.metrics.Accuracy()
for loop in range(1, opt.LOOPS+1):
    if opt.ACQUISITION == 'al_acquisition':
        querry_pool_indices = al_acquisition_func.sample(classifier, fea_d, current_trunl_data, opt.BUDGET)

    '''get querry index in the form of (x, y)'''
    querry_idx = []
    for idx in querry_pool_indices:
        querry_idx.append(current_trunl_idx[idx])

    '''update current_trunl_idx'''
    _tr_unlabeled_idx = []
    for idx in current_trunl_idx:
        if idx not in querry_idx:
            _tr_unlabeled_idx.append(idx)
    current_trunl_idx = _tr_unlabeled_idx

    '''get data/label for next stage training'''
    current_trl_idx = list(current_trl_idx) + list(querry_idx)
    current_trl_data, current_trl_label = get_data(data, label, current_trl_idx)
    current_trunl_data, current_trunl_label = get_data(data, label, current_trunl_idx)
    '''organize current training data'''
    current_trl_data = np.expand_dims(current_trl_data, axis=4)
    current_trl_label = keras.utils.to_categorical(current_trl_label)
    current_trunl_data = np.expand_dims(current_trunl_data, axis=4)
    current_trunl_label = keras.utils.to_categorical(current_trunl_label)
    current_trl_set = tf.data.Dataset.from_tensor_slices((current_trl_data, current_trl_label)).shuffle(len(current_trl_data)).batch(opt.BATCH_SIZE)

    print("Start resuming the", loop, "loops active training (Scheme1) using current labeled training data!")
    for epoch in range(opt.AL_EPOCH):
        for step, (_data, _label) in enumerate(current_trl_set):
            with tf.GradientTape() as tape:
                _fea, _prediction = classifier(_data, training=True)
                _loss = classifier_loss(_label, _prediction)


            grads = tape.gradient(_loss, classifier.trainable_variables)
            optim.apply_gradients(zip(grads, classifier.trainable_variables))

        # al_te_metric.reset_states()
        # for x, y in te_set:
        #     _, _te_pred = classifier(x, training=False)
        #     y = tf.argmax(y, axis=1)
        #     _te_pred = tf.argmax(_te_pred, axis=1)
        #     al_te_metric.update_state(y, _te_pred)
        #
        # print("active testing:", "loop", loop, "epoch", epoch, "acc:", al_te_metric.result().numpy())

    if opt.ACQUISITION == 'al_acquisition' and loop != opt.LOOPS:
        '''freeze the training of classifier and obtain the intermediate features of current labeled data.'''
        current_trl_fea = []
        for step, (_data, _label) in enumerate(current_trl_set):
            currentl_fea, _ = classifier(_data, training=False)
            current_trl_fea.extend(currentl_fea)

        current_trl_fea = np.array(current_trl_fea)
        current_trl_fea_set = tf.data.Dataset.from_tensor_slices(current_trl_fea).shuffle(len(current_trl_fea)).batch(opt.BATCH_SIZE)

        print("Start resuming training GAN using the feature of current labeled training data!")
        for epoch in range(opt.AL_EPOCH):
            for step, _fea in enumerate(current_trl_fea_set):
                if opt.use_MS:
                    disc_loss = train_d_msgan_step(_fea)
                    gen_loss = train_g_msgan_step()
                else:
                    disc_loss = train_d_step(_fea)
                    gen_loss = train_g_step()


"""
Test.
"""
test_func(te_data, te_label)

"""
Generate classification map.
"""
print("Start generating classification map using full data!")
def patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + opt.WINDOW_SIZE)
    width_slice = slice(width_index, width_index + opt.WINDOW_SIZE)
    data_patch = data[height_slice, width_slice, :]

    return data_patch

'''load the original image'''
raw_data, raw_label = load(opt.DATASET)
if opt.use_PCA:
    raw_data, pca = apply_pca(raw_data)
else:
    pass
margin = int((opt.WINDOW_SIZE - 1) / 2)
raw_data_padded = pad_zeros(raw_data, margin=margin)

'''calculate the predicted image'''
pred_map = np.zeros((raw_label.shape[0], raw_label.shape[1]))
for row in range(raw_label.shape[0]):
    for col in range(raw_label.shape[1]):
        target = int(raw_label[row, col])
        if target == 0:
            continue
        else:
            img_patch = patch(raw_data_padded, row, col)
            data_te_img = img_patch.reshape(1,img_patch.shape[0],img_patch.shape[1], img_patch.shape[2], 1).astype('float32')
            _, prediction = classifier.predict(data_te_img)
            prediction = np.argmax(prediction, axis=1)
            pred_map[row][col] = prediction+1

spectral.save_rgb(os.path.join(opt.RESULT, str(opt.DATASET)+"_"+str(opt.ACQUISITION)+"_predictions.jpg"), pred_map.astype(int), colors=spectral.spy_colors)
spectral.save_rgb(os.path.join(opt.RESULT, str(opt.DATASET) + "_groundtruth.jpg"), raw_label, colors=spectral.spy_colors)
