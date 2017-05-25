"""
General-purposed word dictionary

It support importing/exporting from various format and maintain both
word-to-index and index-to-word mappings. Besides mimicking a build-in dict,
it has `batch_get` method and `batch_get_inv` method for frequently usage in
the NLP field.

>>> from worddict import WordDict
>>> wd = WordDict.fromcsv('./data/worddict.txt')

>>> wd['they']
50

>>> wd.batch_get_inv([2, 50, 23, 32, 14, 3])
['<bos>', 'they', 'do', 'not', '!', '<eos>']

>>> wd.batch_get(['<bos>', 'they', 'do', 'not', '!', '<eos>'])
[2, 50, 23, 32, 14, 3]

"""

from collections import abc

class WordDict(abc.MutableMapping):
    def __init__(self, mapping=None):
        self.mapping = {}
        self.inv_mapping = {}

        if mapping is not None:
            self.mapping = {
                k: v for k, v in mapping.items()
                if isinstance(k, str) and isinstance(v, int)
            }
            self.inv_mapping = {v: k for k, v in self.mapping.items()}

    def batch_get(self, keys):
        return [self.mapping[k] if k in self.mapping else 0 for k in keys]

    def batch_get_inv(self, keys):
        return [self.inv_mapping[k] if k in self.inv_mapping else '' for k in keys]

    def __iter__(self):
        return self.mapping.__iter__()

    def __len__(self):
        return len(self.mapping)

    def __delitem__(self, key):
        if isinstance(key, str):
            value = self.mapping[key]
            del self.mapping[key]
            del self.inv_mapping[value]
        elif isinstance(key, int):
            value = self.inv_mapping[key]
            del self.mapping[value]
            del self.inv_mapping[key]
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        """The default behavior is str -> int"""
        return self.mapping[key] if key in self.mapping else 0

    def __setitem__(self, key, value):
        self.mapping[key] = value
        self.inv_mapping[value] = key

    def tocsv(self, path, delimiter=' '):
        import csv
        with open(path, 'w', errors='ignore', newline='') as f:
            writer = csv.writer(f, delimiter=delimiter)
            for k, v in self.mapping.items():
                writer.writerow((k, v))

    def tojson(self, path):
        import json
        with open(path, 'w', errors='ignore') as f:
            json.dump(self.mapping, f)

    @classmethod
    def fromlines(cls, lines, idx_start=0, limit=None):
        from collections import Counter
        counter = Counter()
        for line in lines:
            counter.update(line.split())

        words = counter.most_common(limit)
        _dict = {word: i for i, (word, _) in enumerate(words, start=idx_start)}
        return cls(_dict)

    @classmethod
    def fromcsv(cls, path, delimiter=' '):
        import csv
        with open(path, errors='ignore', newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            _dict = {a: int(b) for a, b, *_ in reader}
            return cls(_dict)

    @classmethod
    def fromjson(cls, path):
        import json
        with open(path, errors='ignore') as f:
            _dict = json.load(f)
            return cls(_dict)
