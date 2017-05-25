from acwgan_model import ACWGAN_model
from condgan_model import conditional_GAN_model
import numpy as np

IMG_SHAPE = (64, 64, 3)
Z_DIM = 100


def main():

    model = conditional_GAN_model(batch_size=5)
    model.create_computing_graph()
    model.initialize_network()
    model.load_model('./cmodel_10501.ckpt-10501')
    
    tag_input = np.load('features.npy')
    for i, feature in enumerate(tag_input):
        model.save_test_img(feature, i)


if __name__ == '__main__':
    main()
