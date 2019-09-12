import os
import subprocess as sp
from collections import namedtuple
from argparse import ArgumentParser


if __name__ != '__main__':
    exit()

parser = ArgumentParser('Bootstrap preprocessing')
parser.add_argument('datadir', help='data dir')
parser.add_argument('--worddict-limit', type=int, help='maximum number of words in output dictionary')
parser.add_argument('--linelen-limit', type=int, help='maximum number of words in each line', default=30)
args = parser.parse_args()

Candidate = namedtuple('Candidate', ['name', 'path', 'datadir'])

data_dir = args.datadir
worddict_limit = args.worddict_limit
linelen_limit = args.linelen_limit
file_dir = os.path.realpath(os.path.dirname(__file__))
scripts = os.listdir(file_dir)

# Get candidate scripts
candidates = []
for d in os.listdir(data_dir):
    if '{}_preprocess.py'.format(d) in scripts:
        print('Found dataset {}.'.format(d))
        script_path = os.path.join(file_dir, '{}_preprocess.py'.format(d))
        dataset_dir = os.path.join(data_dir, d)
        candidates.append(Candidate(d, script_path, dataset_dir))

# Generate wordcount for each dataset
wordcount_command = 'python {0} --data-dir {1} create-word-count'
for cand in candidates:
    command = wordcount_command.format(cand.path, cand.datadir)
    print('* Execute command:', command)
    sp.run(command.split(), stdout=sp.PIPE, check=True)

# Aggregate worddict
worddict_command = 'python {0} --output {1} --limit {2} {3}'.format(
    os.path.join(file_dir, 'aggregate_worddict.py'),
    os.path.join(data_dir, 'worddict.txt'),
    worddict_limit,
    ' '.join([cand.datadir for cand in candidates])
)
print('* Execute command:', worddict_command)
sp.run(worddict_command.split(), stdout=sp.PIPE, check=True)

# Generate linedict for each dataset
linedict_command = 'python {0} --data-dir {1} create-line-dict --worddict {2} --limit {3}'
for cand in candidates:
    command = linedict_command.format(cand.path,
                                      cand.datadir,
                                      os.path.join(data_dir, 'worddict.txt'),
                                      linelen_limit)
    print('* Execute command:', command)
    sp.run(command.split(), stdout=sp.PIPE, check=True)

# Generate convdict for each dataset
convdict_command = 'python {0} --data-dir {1} create-conv-dict'
for cand in candidates:
    command = convdict_command.format(cand.path, cand.datadir)
    print('* Execute command:', command)
    sp.run(command.split(), stdout=sp.PIPE, check=True)

# Aggregate convdict and linedict
worddict_command = 'python {0} --convdict {1} --linedict {2} {3}'.format(
    os.path.join(file_dir, 'aggregate_conv.py'),
    os.path.join(data_dir, 'convdict.txt'),
    os.path.join(data_dir, 'linedict.txt'),
    ' '.join([cand.datadir for cand in candidates])
)
print('* Execute command:', worddict_command)
sp.run(worddict_command.split(), stdout=sp.PIPE, check=True)

