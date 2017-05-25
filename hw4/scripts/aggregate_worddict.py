import json
from argparse import ArgumentParser
from collections import Counter

def main(args):
    limit = args.limit
    counter = Counter()
    for _file in args.files:
        with open(_file) as f:
            counter.update(json.load(f))

    common = counter.most_common(limit)
    _dict = {w: i
             for i, (w, _) in enumerate(common, start=1)}
    _dict['<unk>'] = 0

    with open(args.output, 'w') as f:
        json.dump(_dict, f)

if __name__ == '__main__':
    parser = ArgumentParser('Aggregate multiple wordcount.json into one worddict.json')
    parser.add_argument('output', default='worddict.json', help='output worddict file')
    parser.add_argument(
        '--limit', type=int, default=3000,
        help='maximum number of word in output dictionary')
    parser.add_argument('files', nargs='+', help='input wordcount files')
    args = parser.parse_args()
    main(args)
