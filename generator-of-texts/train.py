import argparse
import sys
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', nargs='?', default=os.getcwd(),
                    help='Directory with texts to train model')
parser.add_argument('--model', nargs='?',
                    type=argparse.FileType('w', encoding="utf-8"),
                    default=sys.stdout,
                    help='Output file')
parser.add_argument('--lc', nargs='?', const=True, default=False,
                    help='To lowercase')

args = parser.parse_args()

ins = vars(args).get('input_dir')
outs = vars(args).get('model')
lc = vars(args).get('lc')


class Model:
    def __init__(self):
        self.words = dict()
        self.frequency = dict()
        self.wordsList = list()
        self.wordsFrequency = list()
        self.newWords = dict()

    def get_word(self, filename):
        file = open(filename, 'r', encoding="utf-8")
        while True:
            s = str()
            while True:
                symb = file.read(1)
                if symb == "":
                    if s != "":
                        break
                    file.close()
                    return
                if s != "" and (symb == " " or symb == "\n"):
                    break
                if symb.isalpha() or symb == '-':
                    if lc:
                        s += symb.lower()
                    else:
                        s += symb
            if s not in self.words:
                self.words[s] = dict()
            yield s

    def get(self):
        for filename in os.listdir(ins):
            full_filename = ins + '/' + filename
            previous_word = None
            for word in self.get_word(full_filename):
                if previous_word is not None:
                    if previous_word not in self.words[word]:
                        self.words[word].setdefault(previous_word, 0)
                    self.words[word][previous_word] += 1
                if word not in self.frequency:
                    self.frequency[word] = 0
                self.frequency[word] += 1
                previous_word = word

    def getLists(self):
        dList = list()
        dFrequency = list()
        dWords = 0
        for key, value in self.frequency.items():
            dWords += value
        for key, value in self.frequency.items():
            dList.append(key)
            dFrequency.append(value / dWords)
        return (dList, dFrequency)

    def process(self):
        self.wordsList, self.wordsFrequency = self.getLists()
        for key, value in self.words.items():
            self.newWords[key] = self.getLists()

    def write(self):
        outs.write(json.dumps([self.wordsList, self.wordsFrequency, self.newWords], ensure_ascii=False))

if __name__ == '__main__':
    model = Model()
    model.get()
    model.process()
    model.write()
