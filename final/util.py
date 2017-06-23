import json
import numpy as np

def load_worddict(filepath):
    with open(filepath, 'r') as f:
        w2id_dict = json.load(f)

    return w2id_dict

def generate_w2vec_txt(embed_arr, id2w_dict, outfile='wordvec.txt'):
    
    with open(outfile, 'w') as f:
        for i, vec in enumerate(embed_arr):
            wvec_list = [id2w_dict[i]]
            wvec_list.extend(['%.5f' % num for num in vec])
            
            wvec_str = ' '.join(wvec_list)
            f.write('%s\n' % wvec_str)
    
