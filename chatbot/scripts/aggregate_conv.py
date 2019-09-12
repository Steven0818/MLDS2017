import os
import json
from queue import deque
from collections import namedtuple
from argparse import ArgumentParser
from tqdm import tqdm
Line = namedtuple('Line', ['idx', 'content'])

def main(args):
    linedict = {}
    convdict = deque()
    for ds in args.datasets:
        with open(os.path.join(ds, 'convdict.json')) as f:
            convs = json.load(f)
        with open(os.path.join(ds, 'linedict.json')) as f:
            lines = json.load(f)
            lines = {int(k): Line(i, lines[k]) for i, k in enumerate(lines, start=len(linedict))}
        for conv in tqdm(convs):
            convdict.append([lines[l].idx for l in conv])

        linedict.update({v.idx: (k, v.content) for k, v in lines.items()})

    with open(args.convdict, 'w') as f:
        for conv in convdict:
            f.write(' '.join(str(c) for c in conv) + '\n')

    with open(args.linedict, 'w') as f:
        for k, v in linedict.items():
            line = '{0} {1} {2}\n'.format(k, v[0], ' '.join(str(c) for c in v[1]))
            f.write(line)

if __name__ == '__main__':
    parser = ArgumentParser('Aggregate multiple wordcount.json into one worddict.json')
    parser.add_argument('--convdict', default='convdict.txt', help='output convdict file')
    parser.add_argument('--linedict', default='linedict.txt', help='output linedict file')
    parser.add_argument('datasets', nargs='+', help='input dataset\'s path')
    args = parser.parse_args()
    main(args)
