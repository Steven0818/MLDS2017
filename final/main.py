import util
from loader import Data
from model import Skipgram_model

vocab_size = 100000
embed_size = 300
window_size = 5
batch_size = 128
sample_num = 100

wiki_corpus = 'data/enwiki-latest-pages-articles.xml.bz2'
wiki_dict = 'data/wiki_en_dict.dict'
wiki_word = 'wiki_en_top_100000_word.pckl'

def main():
    data = Data(wiki_corpus, wiki_dict, wiki_word, vocab_size=vocab_size, window_size=window_size)
    
    skipgram_model = Skipgram_model(vocab_size=30000, embed_size=300, window_size=5, batch_size=128, 
                                    sample_num=100, learning_rate=0.2, learning_decay=0.99)
    
    skipgram_model.create_placeholder()
    skipgram_model.build_graph()
    skipgram_model.build_test_grap()

    skipgram_model.initialize()

    skipgram_model.train(data, epoch=5, batch_size=batch_size)

if __name__ == '__main__':
    main()
