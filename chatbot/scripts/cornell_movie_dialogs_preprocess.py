import os
import sys
import ast
import json
from collections import Counter
import fire
from tqdm import tqdm
from preprocess import line_preprocess

PROJECT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_DIR)
# pylint: disable=C0413
from worddict import WordDict   # pylint: disable=E0401

class Main():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.conversation_fpath = os.path.join(data_dir, 'movie_conversations.txt')
        self.line_fpath = os.path.join(data_dir, 'movie_lines.txt')

    def _load_lines(self):
        line_dict = {}
        for line in tqdm(open(self.line_fpath, errors='ignore')):
            index, *_, content = line.strip().split(' +++$+++ ')
            if not index.startswith('L'):
                raise ValueError('Line not starts with "L": {}' % index)

            line_dict[int(index[1:])] = line_preprocess(content)
        return line_dict

    def _load_conversations(self):
        convs = []
        for line in open(self.conversation_fpath, errors='ignore'):
            *_, line_literal = line.split(' +++$+++ ')
            value = ast.literal_eval(line_literal)
            if (not isinstance(value, list) or
                    not all(isinstance(v, str) for v in value) or
                    not all(v.startswith('L') for v in value)
               ):
                raise ValueError('Line literal not in correct format: {}' % line_literal)

            convs.append([int(v[1:]) for v in value])
        return convs

    def create_word_count(self, output='wordcount.json', limit=3000):
        lines = self._load_lines()

        counter = Counter()
        for line in lines.values():
            counter.update(line.split())

        _dict = {k:c for k, c in counter.most_common(limit)}
        fpath = os.path.join(self.data_dir, output)
        with open(fpath, 'w') as f:
            json.dump(_dict, f)

    def create_line_dict(self, worddict='worddict.txt', output='linedict.json', limit=30):
        lines = self._load_lines()

        outpath = os.path.join(self.data_dir, output)
        indict = WordDict.fromcsv(worddict)

        _dict = {i: indict.batch_get(line.split()) for i, line in lines.items()}
        _dict2 = {i: line for i, line in _dict.items() if len(line) <= limit}

        with open(outpath, 'w') as f:
            json.dump(_dict2, f)

    def create_conv_dict(self, linedict='linedict.json', output='convdict.json'):
        convs = self._load_conversations()
        linedict = json.load(open(os.path.join(self.data_dir, linedict)))

        convs = [conv for conv in tqdm(convs)
                 if all(str(line) in linedict for line in conv)]

        fpath = os.path.join(self.data_dir, output)
        with open(fpath, 'w') as f:
            json.dump(convs, f)


    def __call__(self):
        pass

if __name__ == '__main__':
    fire.Fire(Main)
