import sys
import pandas
import word2vec
import json

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=''):
        if desc != '':
            print(desc)
        for obj in iterable:
            yield obj

def preprocess_test(dic, qpath):
    dic['_____'] = -1

    questions = pandas.read_csv(qpath)
    questions.columns = ['id', 'question', 'a', 'b', 'c', 'd', 'e']

    def get_idx(word):
        if word in dic:
            return dic[word]
        else:
            return 0


    def get_sentence_idx(sentence):
        words = sentence.lower().replace(',', ' , ').replace('.', '').split()
        return [dic[word] if word in dic else 0 for word in words]

    ret = []
    for i, q in tqdm(questions.iterrows(), desc='Looking up'):
        a = {}
        a['answer'] = [get_idx(x) for x in [q.a, q.b, q.c, q.d, q.e]]
        a['sentence'] = get_sentence_idx(q.question)
        ret.append(a)
    return ret

def print_usage():
    print('Usage:')
    print('\tpython3 preprocess_test.py <input-dict> <input-csv> <output-json>')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print_usage()
        exit()

    dic = json.load(open(sys.argv[1]))

    json.dump(preprocess_test(dic, sys.argv[2]), open(sys.argv[3], 'w'))
