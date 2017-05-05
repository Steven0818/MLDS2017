from gan_model import GAN_model
from input import DataLoader

IMG_SHAPE = (96,96,3)
Z_DIM = 100

def main():
    dataLoader = DataLoader("data/faces", IMG_SHAPE)
    
    model = GAN_model(z_dim=Z_DIM, batch_size=100, learning_rate=0.0002, img_shape=IMG_SHAPE)

    model.create_computing_graph()
    model.initialize_network()
    model.train_model(dataLoader, 30)
    

if __name__ == '__main__':
    main()
