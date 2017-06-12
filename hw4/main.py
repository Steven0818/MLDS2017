from loader import Data
from model import grl_model
import tensorflow as tf

class GRLConfig(object):
    beam_size = 5
    learning_rate = 0.001
    learning_rate_decay_factor = 0.999
    max_gradient_norm = 1.0
    batch_size = 40
    emb_dim = 500
    num_layers = 2
    vocab_size = 7000
    tensorboard_dir = "./tensorboard/grl_log/"
    name_loss = "grl_loss"
    pre_name_loss = "pre_rl_loss"
    max_train_data_size = 0
    steps_per_checkpoint = 200
    buckets =        [(10, 10), (15, 10), (20, 15), (40, 25), (50,30)]#, (80,50), (100,50)]
    #buckets_concat = [(5, 10), (10, 15), (20, 25), (40, 50), (100, 50)]

def main():
    grl_config = GRLConfig()
    cornell_data = Data('./scripts/', [(10, 10), (15, 10), (20, 15), (40, 25), (50,30)], batch_size=40)
    model = grl_model(grl_config, num_samples=512, forward=False, beam_search=False, dtype=tf.float32)
    model.create_placeholder()
    model.build_training_graph('grl_model')
    model.initialize_network()

    model.train(cornell_data, iteration=30000)

if __name__ == "__main__":
    main()