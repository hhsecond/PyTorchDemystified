import os


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        return f.read()


def vocab_gen(text, outset):
    for sent in text:
        for word in sent.split():
            outset.add(word)
    return dict(enumerate(list(outset)))


def vocab_to_ids(text, vocab):
    pass


def whole_text_ids(text, word2id):
    out = []
    for sent in text:
        for word in sent.split():
            out.append(word2id[word])


en_text_path = 'small_vocab_en'
fr_text_path = 'small_vocab_fr'

en_text = load_data(en_text_path).split('\n')
fr_text = load_data(fr_text_path).split('\n')
for i in range(len(fr_text)):
    fr_text[i] += ' <EOS>'

# passing dummy python set coz 'new' is common for both eng and french
en_vocab = vocab_gen(en_text, {'new'})
fr_vocab = vocab_gen(fr_text, {'new'})

en_text_ids = whole_text_ids(en_text, en_words2id)
fr_text_ids = whole_text_ids(fr_text, fr_words2id)
