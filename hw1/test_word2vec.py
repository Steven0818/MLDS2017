import word2vec

# Load container model
# This .bin file is downloaded from
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
model = word2vec.load_word2vec_model('./data/word2vec/GoogleNews-vectors-negative300.bin')

assert model['the'].shape == (1, 300)
assert model[['two', 'words']].shape == (2, 300)

# The subset model can be create by procedure/word2vec_make_subset.py
model2 = word2vec.load_indexed_model('./data/subset.bin')

assert model2[2].shape == (1, 300)   # 2 is mapped to 'the'
assert model2[1, 2].shape == (2, 300)   # 2 is mapped to 'the'
