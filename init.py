import pandas as pd

def init():
    test = pd.read_csv('liar_dataset/test.tsv', sep = '\t')
    train = pd.read_csv('liar_dataset/train.tsv', sep = '\t')
    valid = pd.read_csv('liar_dataset/valid.tsv', sep = '\t')

    return train, valid, test





