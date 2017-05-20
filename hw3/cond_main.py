from acwgan_model import ACWGAN_model
from condgan_model import conditional_WGAN_model
from condition_input import DataLoader

IMG_SHAPE = (64,64,3)
Z_DIM = 100

def main():
    dataLoader = DataLoader("data/single_feat_faces", IMG_SHAPE, 'tags.json')
    
    model = ACWGAN_model(z_dim=Z_DIM, batch_size=100, learning_rate=0.0002, img_shape=IMG_SHAPE, optimizer_name='RMSProp', 
                         eyes_dim = 11, hair_dim = 11, clip_value=(-0.01,0.01), iter_ratio=6)
    
    #model = conditional_WGAN_model(z_dim=Z_DIM, batch_size=100, learning_rate=0.0002, img_shape=IMG_SHAPE, optimizer_name='RMSProp', 
    #                               tag_dim = 22, tag_embed_dim = 50, clip_value=(-0.01,0.01), iter_ratio=6)
    
    model.create_computing_graph()
    model.initialize_network()
    model.train_model(dataLoader, 100000)
    

if __name__ == '__main__':
    main()
