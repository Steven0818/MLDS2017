#!/bin/bash
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt -P ./skipthoughts/models/
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy -P ./skipthoughts/models/
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy -P ./skipthoughts/models/
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz -P ./skipthoughts/models/
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl -P ./skipthoughts/models/
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz -P ./skipthoughts/models/
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl -P ./skipthoughts/models/

wget https://www.dropbox.com/s/0rbuifmkv716a2e/cmodel_11001.ckpt-11001.data-00000-of-00001?dl=0 -O "cmodel_11001.ckpt-11001.data-00000-of-00001"
wget https://www.dropbox.com/s/357ailxwps2vioo/cmodel_11001.ckpt-11001.index?dl=0 -O "cmodel_11001.ckpt-11001.index"
wget https://www.dropbox.com/s/58z6nrtz01gr4qd/cmodel_11001.ckpt-11001.meta?dl=0 -O "cmodel_11001.ckpt-11001.meta"

python2 skipthoughts/test_preprocess.py $1
python3 generate.py