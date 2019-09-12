wget https://www.dropbox.com/s/u0o88vjnvxodzbp/dict_rev.json?dl=0 -O "dict_rev.json"
wget https://www.dropbox.com/s/t2lkwqhufovuv2y/model_30000.ckpt.data-00000-of-00001?dl=0 -O "model_30000.ckpt.data-00000-of-00001"
wget https://www.dropbox.com/s/fbc4q9emm5apbk6/model_30000.ckpt.index?dl=0 -O "model_30000.ckpt.index"
wget https://www.dropbox.com/s/eattj4i281iobcu/model_30000.ckpt.meta?dl=0 -O "model_30000.ckpt.meta"
python3 test_main.py $1 $2