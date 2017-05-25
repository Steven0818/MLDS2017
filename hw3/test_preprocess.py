"""
run in python2
require skip-thoughts
"""
import sys
import re
import skipthoughts.skipthoughts
import numpy as np


def main():
    with open(sys.argv[1]) as f:
        features = f.readlines()
    features = [re.match('[0-9]*,(.*)', x).groups()[0] for x in features]
    print(features)
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    ret = encoder.encode(features)
    np.save('features.npy', ret)


if __name__ == '__main__':
    main()
