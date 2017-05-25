"""
transfer tags to embedding
"""

import skipthoughts
import json
import numpy as np
import random

model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

#11
eyes_color_list = ['gray', 'aqua', 'orange', 'red', 'blue', 'black', 'pink', 'green', 'brown', 'purple', 'yellow']
#11
hair_color_list = ['gray', 'aqua', 'pink', 'white', 'red', 'purple', 'blue', 'black', 'green', 'brown', 'orange']

def main(file_path='tags.json'):
    def extract_feature(data):
        f_str = ''
        w_str = ''
        for eyes in data['eyes']:
            f_str += eyes_color_list[eyes] + ' ' + 'eyes '
        for hair in data['hair']:
            f_str += hair_color_list[hair] + ' ' + 'hair '
        w_str += random.choice([x for i, x in enumerate(hair_color_list) if i not in data['hair']]) + ' ' + 'hair '
        w_str += random.choice([x for i, x in enumerate(eyes_color_list) if i not in data['eyes']]) + ' ' + 'eyes '
        return f_str, w_str

    tag = json.load(open(file_path))
    ret = {}
    w_ret = {}
    index = []
    features = []
    w_features = []
    for k, v in tag.iteritems():
        feature, w_feature = extract_feature(v)
        if feature:
            index.append(k)
            features.append(feature)
            w_features.append(w_feature)
    print 'start embedding'
    embed_features = encoder.encode(features)
    embed_w_features = encoder.encode(w_features)
    print 'end embedding'

    print embed_features.shape
    print embed_w_features.shape
    json.dump(index, open('feature_order.json', 'w'))
    
    np.save('tags.npy',embed_features)
    np.save('w_tags.npy',embed_w_features)

if __name__ == '__main__':
    main()
    
    
