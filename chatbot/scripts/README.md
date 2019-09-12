# Scripts Usage

Every dataset will have a corresponding `{name}_preprocess.py`. Each script
will use `fire` as argument parser. The command-line options will be almost the
same for these scripts.

To manually execute preprocessing procedure, one should first generate
`wordconut.json` for each dataset using the preprocessing script. The `wordcount.json`
file records word frequency. After generating these files, call
`aggregate_worddict.py` to generate the final word dictionary `worddict.txt`, which can
later be loaded using `worddict.py` in the code base. Then, use the word dictionary
file to generate `linedict.json` for each dataset and use the same script to
generate `convdict.json`. Finally, call `aggregate_conv.py` to generate the
main `convdict.txt` and `linedict.txt`. Only by using this procedure can we
have the same word dictionary for all the datasets.

We provide `bootstrap.py` for the above routine. After running this script,
`worddict.txt`, `linedict.txt`, and `convdict.txt` will be generated under
`<data-dir>`. However, before runnning this script, make sure there is a data
folder which containing dataset folders, and each dataset folder should have a
corresponding `{name}_preprocess.py` in the script folder. As for how to create
your own preprocessing script, copy the existing `{name}_preprocess.py` and
start hacking.

Finally, the core preprocess logic for each line is written in `preprocess.py`
If there is any change in the preprocessing scripts, re-run `bootstrap.py` to
overwrite the old files.

# File Formats

## worddict.txt

Each line contains the target word and corresponding int mapping. It's recommended
to use `WordDict.fromcsv` to load this file. See `worddict.py` for example usage.

## linedict.txt
Each line is of the following format:

```
<line-id> <dataset-local-id> <word1> <word2> ...
```

It can be easily loaded using the following code:

```python
lines = {}
with open('linedict.txt') as f:
    for line in f:
        i, _, *words = line.split()
        lines[int(i)] = words
```

## convdict.txt
Each line is of the following format:

```
<line1> <line2> ...
```

It can be easily loaded using the following code:

```python
convs = []
with open('convdict.txt') as f:
    convs.append(line.split() for line in f)
```
