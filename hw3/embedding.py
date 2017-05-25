import skipthoughts.skipthoughts as st
import numpy as np
import json

def create_embedding_npy(json_file='', ):
    
    model = st.load_model()

    eyes_color_list = ['gray', 'aqua', 'orange', 'red', 'blue', 'black', 'pink', 'green', 'brown', 'purple', 'yellow']
    hair_color_list = ['gray', 'aqua', 'pink', 'white', 'red', 'purple', 'blue', 'black', 'green', 'brown', 'orange']

    fidx2arridx_dict = {}

    jobj = json.load(open(json_file, 'r'))

    tag_strs = [] 
    count = 0
    for fidx, color_d in jobj.items(): 
        if len(color_d['eyes']) == 1 and len(color_d['hair']) == 1:
            eyes_color = eyes_color_list[color_d['eyes'][0]]
            hair_color = hair_color_list[color_d['hair'][0]] 
            tag_str = ' '.join([hair_color, 'hair', eyes_color, 'eyes'])
            tag_strs.append(tag_str) 

            fidx2arridx_dict[fidx] = count 
            count += 1

    tag_embeddings = st.encode(model, tag_strs)

    print tag_embeddings.shape
    print len(fidx2arridx_dict)

    with open('fidx2arridx.json', 'w') as f:
        json.dump(fidx2arridx_dict, f)
    
    np.save('tags_embedding.npy', tag_embeddings)