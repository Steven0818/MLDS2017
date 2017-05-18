import json
from collections import Counter

def create_tags_json(infile='data/tags_clean.csv', outfile='tags.json'):
    tags_id2idx_dict = {}

    #11
    eyes_color_list = ['gray', 'aqua', 'orange', 'red', 'blue', 'black', 'pink', 'green', 'brown', 'purple', 'yellow']
    #11
    hair_color_list = ['gray', 'aqua', 'pink', 'white', 'red', 'purple', 'blue', 'black', 'green', 'brown', 'orange']

    eyes_color_dict = {}
    hair_color_dict = {}

    for i in range(len(eyes_color_list)):
        eyes_color_dict[eyes_color_list[i]] = i
    
    for i in range(len(hair_color_list)):
        hair_color_dict[hair_color_list[i]] = i

    with open(infile, 'r') as f:
        lines = f.readlines()
        for l in lines:
            img_id, img_cap = l.strip().split(',')
            img_tags = [t.split(':')[0].strip() for t in img_cap.split('\t')]
            eyes_tags = [t.split()[0] for t in img_tags if t.find('eyes') != -1]
            hair_tags = [t.split()[0] for t in img_tags if t.find('hair') != -1 and t.find('michairu') == -1]

            eyes_id_tags = [eyes_color_dict[t] for t in eyes_tags if t in eyes_color_dict]
            hair_id_tags = [hair_color_dict[t] for t in hair_tags if t in hair_color_dict]

            tags_id2idx_dict[img_id] = {'eyes': eyes_id_tags, 'hair': hair_id_tags}

    with open(outfile, 'w') as f:
        json.dump(tags_id2idx_dict, f)

def check_tags_type(tags_file='data/tags_clean.csv'):
    eyes_counter = Counter()
    hair_counter = Counter()
    
    with open(tags_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            img_id, img_cap = l.strip().split(',')
            img_tags = [t.split(':')[0].strip() for t in img_cap.split('\t')]
            eyes_tags = [t.split()[0] for t in img_tags if t.find('eyes') != -1]
            hair_tags = [t.split()[0] for t in img_tags if t.find('hair') != -1 and t.find('michairu') == -1]

            eyes_counter.update(eyes_tags)
            hair_counter.update(hair_tags)

    print('There are %d kinds of Eyes.' % len(eyes_counter))
    print('There are %d kinds of Hair.' % len(hair_counter))

    eyes_idx_dict = {}
    hair_idx_dict = {}

    idx = 0
    for k, _ in eyes_counter.items():
        eyes_idx_dict[k] = idx
        idx += 1
    
    idx = 0
    for k, _ in hair_counter.items():
        hair_idx_dict[k] = idx
        idx += 1
    
    print(eyes_idx_dict)
    print(hair_idx_dict )


    
    
    
                                        