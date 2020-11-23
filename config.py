import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('-GPU', type=int, default=0, help='which gpu device to use')
        self.parser.add_argument('-DATASET', default='Indian', choices=['Indian', 'PaviaU'],
                                 help='which data set for experiment')
        self.parser.add_argument('-N_CLS', type=int, default=16, choices=[16, 9],
                                 help='how many class in the training data')
        self.parser.add_argument('-CHANNEL', type=int, default=30, help='input channels')
        self.parser.add_argument('-WINDOW_SIZE', type=int, default=25, help='size of training/testing patches')
        self.parser.add_argument('-DIM_Z', type=int, default=100, help='dimensionality of noise to generate feature')
        self.parser.add_argument('-INIT_EPOCH', type=int, default=45,
                                 help='how many epochs to train classifier with initial labeled data')
        self.parser.add_argument('-INIT_GAN_EPOCH', type=int, default=45,
                                 help='how many epochs to train gan with initial labeled features')
        self.parser.add_argument('-AL_EPOCH', type=int, default=45,
                                 help='how many epochs to train for active learning loop')
        self.parser.add_argument('-INIT_N_L', type=int, default=5, help='initial number of labeled data per class')
        self.parser.add_argument('-INIT_N_UNL', type=int, default=1000, help='number of data in initial unlabeled pool')
        self.parser.add_argument('-LOOPS', type=int, default=5, help='number of active learning loops')
        self.parser.add_argument('-BUDGET', type=int, default=34, help='budget of active learning')
        self.parser.add_argument('-ACQUISITION', default='al_acquisition', choices=['random', 'least_confidence', 'entropy', 'bald'],
                                 help='which acquisition heuristic to use')
        self.parser.add_argument('-BATCH_SIZE', type=int, default=125, help='training batch size')
        self.parser.add_argument('-DR_RATE', type=float, default=.4, help='dropout rate')
        self.parser.add_argument('-LR', type=float, default=1e-3, help='classifier learning rate')
        self.parser.add_argument('-GAN_LR', type=float, default=0.0002, help='gan learning rate')
        self.parser.add_argument('-DECAY', type=float, default=1e-06, help='decay rate')
        self.parser.add_argument('-use_MS', type=bool, default=True,
                                 help='use mode seeking loss or not when training gan')
        self.parser.add_argument('-use_PCA', type=bool, default=False,
                                 help='use PCA or not')
        self.parser.add_argument('-use_SuperPCA', type=bool, default=True,
                                 help='use SuperPCA or not')
        self.parser.add_argument('-DATA_DIR', default='./dataset/', help='directory to load data')
        self.parser.add_argument('-RESULT', default='./result/', help='directory to save results')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        return self.opt
