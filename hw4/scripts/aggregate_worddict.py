import os
import json
from argparse import ArgumentParser
from collections import Counter

def main(args):
    limit = args.limit
    counter = Counter()
    for ds in args.datasets:
        with open(os.path.join(ds, 'wordcount.json')) as f:
            counter.update(json.load(f))

    common = counter.most_common(limit)
    _dict = {w: i
             for i, (w, _) in enumerate(common, start=2)}
    _dict['<unk>'] = 0
    _dict['<pad>'] = 1

    with open(args.output, 'w') as f:
        for k, v in _dict.items():
            f.write('{0} {1}\n'.format(k, v))

if __name__ == '__main__':
    parser = ArgumentParser('Aggregate multiple wordcount.json into one word dict file')
    parser.add_argument('--output', default='worddict.txt', help='output worddict file')
    parser.add_argument(
        '--limit', type=int, default=5000,
        help='maximum number of word in output dictionary')
    parser.add_argument('datasets', nargs='+', help='input dataset\'s path')
    args = parser.parse_args()
    main(args)
