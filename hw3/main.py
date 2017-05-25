from gan_model import GAN_model, WGAN_model2
from input import DataLoader

IMG_SHAPE = (64,64,3)
Z_DIM = 100

def main():
    dataLoader = DataLoader("data/faces", IMG_SHAPE)
    
    model = WGAN_model2(z_dim=Z_DIM, batch_size=100, learning_rate=0.0002, img_shape=IMG_SHAPE, optimizer_name='RMSProp', clip_value=(-0.01,0.01), iter_ratio=5)

    model.create_computing_graph()
    model.initialize_network()
    model.train_model(dataLoader, 100000)
    

if __name__ == '__main__':
    main()
