#!/bin/bash
wget "https://www.dropbox.com/s/rbwgirx9xb0nnbk/model_50000.ckpt.data-00000-of-00001?dl=0" -O "model_50000.ckpt.data-00000-of-00001"
wget "https://www.dropbox.com/s/h05clz5izmtnuw7/model_50000.ckpt.index?dl=0" -O "model_50000.ckpt.index"
wget "https://www.dropbox.com/s/5wnwntaw7fj1l8m/model_50000.ckpt.meta?dl=0" -O "model_50000.ckpt.meta"
wget "https://www.dropbox.com/s/lexbsqnyfv6dhif/test.json?dl=0"  -O test.json
python3 test_main.py $2 $1
