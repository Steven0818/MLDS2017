"""
run in python2
require skip-thoughts
"""
import sys
import re
from skipthoughts import load_model, Encoder
import numpy as np


def main():
    with open(sys.argv[1]) as f:
        features = f.readlines()
    features = [re.match('[0-9]*,(.*)', x).groups()[0] for x in features]
    print(features)
    model = load_model()
    encoder = Encoder(model)
    ret = encoder.encode(features)
    np.save('features.npy', ret)


if __name__ == '__main__':
    main()
