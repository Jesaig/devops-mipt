import numpy as np
import argparse
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', nargs='?',
                    type=argparse.FileType('r', encoding="utf-8"),
                    default=sys.stdin,
                    help='Model file')
parser.add_argument('--output', nargs='?',
                    type=argparse.FileType('w', encoding="utf-8"),
                    default=sys.stdout,
                    help='Output file')
parser.add_argument('--length', nargs='?',
                    type=int,
                    default=100,
                    help='Generated text length (in words)')
parser.add_argument('--seed', nargs='?',
                    type=str,
                    default=None,
                    help='First word')

args = parser.parse_args()

ins = vars(args).get('model')
outs = vars(args).get('output')


class ModelGenerator:
    def __init__(self):
        data = json.load(ins)
        self.wordsList = data[0]
        self.wordsFrequency = data[1]
        self.newWords = data[2]

    def get_first_word(self):
        return np.random.choice(self.wordsList, 1, self.wordsFrequency)[0]

    def get_word(self, word=-1):
        if word not in self.newWords or len(self.newWords[word][0]) == 0:
            return self.get_first_word()
        return np.random.choice(self.newWords[word][0], 1,
                                self.newWords[word][1])[0]

    def gen_text(self, length, first_word):
        if first_word is not None:
            length -= 1
            outs.write("%s " % first_word)
        word = self.get_word(first_word)
        outs.write("%s " % word)
        for i in range(length - 1):
            word = self.get_word(word)
            outs.write("%s " % word)


if __name__ == "__main__":
    ModelGenerator().gen_text(vars(args).get('length'),
                              vars(args).get('seed'))
