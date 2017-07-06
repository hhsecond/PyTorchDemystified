import os
from collections import namedtuple
import pickle
import numpy as np

data = namedtuple('data', 'id2vocab vocab2id text text_as_ids')


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def vocab2ids(text, outset):
    for sent in text:
        for word in sent.split():
            outset.add(word)
    return dict(enumerate(outset))


def whole_text_ids(text, word2id):
    out = []
    for sent in text:
        out_sent = []
        for word in sent.split():
            out_sent.append(word2id[word])
        out.append(out_sent)
    return out


def get_data():
    en_text_path = 'data/small_vocab_en'
    fr_text_path = 'data/small_vocab_fr'

    en_text = load_data(en_text_path).lower().split('\n')
    fr_text = load_data(fr_text_path).lower().split('\n')
    for i in range(len(fr_text)):
        fr_text[i] += ' <EOS>'

    # passing dummy python set coz 'new' is common for both eng and french
    en_id2vocab = vocab2ids(en_text, {'new'})
    fr_id2vocab = vocab2ids(fr_text, {'new'})
    fr_id2vocab[len(fr_id2vocab)] = '<GO>'
    fr_id2vocab[len(fr_id2vocab)] = '<PAD>'
    fr_id2vocab[len(fr_id2vocab)] = '<UNK>'
    en_id2vocab[len(en_id2vocab)] = '<PAD>'
    en_id2vocab[len(en_id2vocab)] = '<UNK>'

    en_vocab2id = {v: k for k, v in en_id2vocab.items()}
    fr_vocab2id = {v: k for k, v in fr_id2vocab.items()}

    en_text_as_ids = whole_text_ids(en_text, en_vocab2id)
    fr_text_as_ids = whole_text_ids(fr_text, fr_vocab2id)

    # assertion check
    assert len(en_text) == len(en_text_as_ids)
    assert len(fr_text) == len(fr_text_as_ids)
    en = data(en_id2vocab, en_vocab2id, en_text, en_text_as_ids)
    fr = data(fr_id2vocab, fr_vocab2id, fr_text, fr_text_as_ids)
    with open('preprocess.p', 'wb') as out_file:
        pickle.dump((
            (en_text, fr_text),
            (en_vocab2id, fr_vocab2id),
            (en_id2vocab, fr_id2vocab)), out_file)
    return en, fr


def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0, 0), (0, max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0, 0), (0, max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))
