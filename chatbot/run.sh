#!/bin/bash


wget https://www.dropbox.com/s/5hl5gcztywtna3g/movie_subtitle.model-118000.data-00000-of-00001?dl=0 -O "./grl_data/movie_subtitle.model-118000.data-00000-of-00001"
wget https://www.dropbox.com/s/m4dlhql21vb1du4/movie_subtitle.model-118000.index?dl=0 -O "./grl_data/movie_subtitle.model-118000.index"
wget https://www.dropbox.com/s/1km0wp4ebx2i0y8/movie_subtitle.model-118000.meta?dl=0 -O "./grl_data/movie_subtitle.model-118000.meta"
wget https://www.dropbox.com/s/kx2jruxw0auou5z/movie_subtitle.model-127200.data-00000-of-00001?dl=0 -O "./grl_data/movie_subtitle.model-127200.data-00000-of-00001"
wget https://www.dropbox.com/s/ksippveju5welj9/movie_subtitle.model-127200.index?dl=0 -O "./grl_data/movie_subtitle.model-127200.index"
wget https://www.dropbox.com/s/qfr473qz5hk3u4i/movie_subtitle.model-127200.meta?dl=0 -O "./grl_data/movie_subtitle.model-127200.meta"
wget https://www.dropbox.com/s/5qn9axujwj03hxe/vocab25000.all?dl=0 -O "./grl_data/vocab25000.all"

python3 grl_train.py $1 $2 $3