from acwgan_model import ACWGAN_model
from condgan_model import conditional_GAN_model
from condition_input import DataLoader
import numpy as np

IMG_SHAPE = (64, 64, 3)
Z_DIM = 100


def main():
    dataLoader = DataLoader("data/single_feat_faces",
                            IMG_SHAPE, 'tags.npy', 'w_tags.npy', 'order.json')

    model = conditional_GAN_model()
    model.create_computing_graph()
    model.initialize_network()
    model.load_model('./cmodel/cmodel_46001.ckpt-46001')
    # model.test_loader(dataLoader)
    
    tag_input = np.load('tags.npy')
    w_input = np.load('test1.npy')
    test_input = tag_input[0:100]
    for i in range(100):
        test_input[i] = tag_input[i]
    model.test_single_feature(test_input)
    # batch_gen = dataLoader.batch_generator(batch_size=100)
    # for batch_imgs, correct_tag, wrong_tag in batch_gen:
    #     print(correct_tag.shape)
    #     model.test_single_feature(correct_tag)
    #     break


if __name__ == '__main__':
    main()
