import util
from loader import WikiAllData, Wiki9Data
from model import Word2vec_model

#vocab_size = 100000
embed_size = 300
window_size = 5
batch_size = 256
sample_num = 256
model_type = 'skipgram'

#wiki_corpus = 'data/enwiki-latest-pages-articles.xml.bz2'
wiki_corpus = 'data/wiki9.txt'
wiki_dict = 'data/wiki_en_dict.dict'
#wiki_word = 'wiki_en_top_100000_word.pckl'
wiki_word = 'worddict.json'
feqfile = 'wordfeq.json'

def main():
    w2id_dict = util.load_worddict(wiki_word)
    wid2feq_dict = util.load_wordfeq(feqfile)
    vocab_size = len(w2id_dict)
    #data = WikiAllData(wiki_corpus, wiki_dict, wiki_word, vocab_size=vocab_size, window_size=window_size)
    data = Wiki9Data(wiki_corpus, w2id_dict, wid2feq_dict, window_size=window_size)

    skipgram_model = Word2vec_model(vocab_size=vocab_size, embed_size=embed_size, window_size=window_size, batch_size=batch_size, 
                                    sample_num=sample_num, learning_rate=0.05, learning_decay=0.998)
    
    skipgram_model.create_placeholder()
    skipgram_model.build_graph()
    skipgram_model.build_test_grap()
    skipgram_model.initialize(reload=True)
    skipgram_model.train(data, model_type=model_type, epoch=20, batch_size=batch_size)

    embed_arr = skipgram_model.get_embedding_layer()
    
    id2w_dict = {v:k for k, v in w2id_dict.items()}
    
    util.generate_w2vec_txt(embed_arr, id2w_dict)


if __name__ == '__main__':
    main()
