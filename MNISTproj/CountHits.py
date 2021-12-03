import os, sys
from keras.datasets import mnist

def main():
    hits = 0

    (train_data, train_labels), (test_data, test_labels) \
      = mnist.load_data()

    for right_label, file_label in zip(test_labels, sys.stdin):
        if right_label == int(file_label):
            hits += 1

    print("{}".format(hits))

main()
