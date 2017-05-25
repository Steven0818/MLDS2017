# Scripts Usage

Every dataset will have a conresponding `{name}_preprocess.py`. Each script
will use `fire` as argument parser. The commandline options will be almost the
same for every dataset. If manually executing the preprocessing procedure, one
should first generate `wordconut.json` and `convdict.json` for each dataset
using the scripts. The `wordcount.json` file records word frequency, while
`convdict.json` contains a list of lists of line ids.

After generating these files, call `aggregate_worddict.json` to generate the
final word dictionary, which can later be loaded using `worddict.py` in the
code base. And finally, use the word dictionary file to generate
`linedict.json` for each dataset. Only by using this procedure can we have
the same word dictionary for all the datasets.
