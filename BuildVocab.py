import nltk
import pickle
import os
from Config             import config
from collections        import Counter
from pycocotools.coco   import COCO
from utils              import absolute_dir_wrapper


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))

    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def build_and_save_vocab():
    vocab_path = absolute_dir_wrapper(config.get('vocab_path'))
    exists = os.path.isfile(vocab_path)

    if not exists or config.get('overwrite'):
        vocab = build_vocab(json=absolute_dir_wrapper(config.get('train_caption_path')), threshold=config.get('threshold', 0))
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print("Total vocabulary size: {}".format(len(vocab)))
        print("Saved the vocabulary to '{}'".format(vocab_path))

    else:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocabulary is loaded".format(vocab_path))

    return vocab
