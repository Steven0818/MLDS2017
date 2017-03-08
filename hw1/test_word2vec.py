import word2vec

# Load container model
# This .bin file is downloaded from
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
model = word2vec.load_model('./data/word2vec/GoogleNews-vectors-negative300.bin')

assert model['the'].shape == (1, 300)
assert model[['two', 'words']].shape == (2, 300)
