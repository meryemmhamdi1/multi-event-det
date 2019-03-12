try:
    from pycorenlp import StanfordCoreNLP
except:
    pass
from subprocess import call
import numpy as np
from get_args import *
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


def get_iso_lang_abbreviation():
    iso_lang_dict = {}
    lang_iso_dict = {}
    with open("iso_lang_abbr.txt") as file:
        lines = file.read().splitlines()
        for line in lines:
            lang_iso_dict.update({line.split(":")[0]:line.split(":")[1]})
            iso_lang_dict.update({line.split(":")[1]:line.split(":")[0]})
    return iso_lang_dict, lang_iso_dict


class DependencyParser(object):
    def __init__(self, sent):
        self.sent = sent
        self.stanford = StanfordCoreNLP('http://localhost:9001')
        self.properties = {'annotators': 'tokenize,ssplit,pos,depparse,parse', 'outputFormat': 'json'}
        #call(["java -mx4g -cp '*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 15000"])

    def find_dep_words_pos_offsets(self, sent):
        output = self.stanford.annotate(sent, properties=self.properties)
        penn_treebank = output['sentences'][0]['parse'].replace("\n", "")
        triples = []
        for part in output['sentences'][0]['enhancedPlusPlusDependencies']:
            triples.append(part['dep']+"/dep="+str(part['dependent']-1)+"/gov="+str(part['governor']-1))

        words = []
        words_dict = {}
        pos_tags = []
        offset_start_dic = {}
        offset_end_dic = {}
        for i, word in enumerate(output['sentences'][0]['tokens']):
            words.append(word["word"])
            pos_tags.append(word["pos"])
            offset_start_dic.update({word["characterOffsetBegin"]: word["index"]-1})
            offset_end_dic.update({word["characterOffsetEnd"]-1: word["index"]})
            words_dict.update({word["index"]-1: word["word"]})

        return penn_treebank, triples, words, pos_tags, offset_start_dic, offset_end_dic, words_dict


class Embeddings(object):
    def __init__(self, emb_filename, emb_type, vocab, dim_word):
        self.emb_filename = emb_filename
        self.dim = dim_word
        self.trimmed_filename = ".".join(emb_filename.split(".")[:-1]) + "_trimmed.npz"
        if emb_type == "fasttext" or emb_type == "glove":
            self.embed_vocab = self.get_emb_vocab()
        else:
            self.embed_vocab = self.get_multi_emb_vocab()
        if not os.path.isfile(self.trimmed_filename):
            if emb_type == "fasttext":
                self.export_trimmed_fasttext_vectors(vocab)
                self.embed_vocab = self.get_emb_vocab()
            elif emb_type == "glove":
                self.export_trimmed_glove_vectors(vocab)
                self.embed_vocab = self.get_emb_vocab()
            else:
                self.export_trimmed_multi_vectors(vocab)
                self.embed_vocab = self.get_multi_emb_vocab()

    def export_trimmed_fasttext_vectors(self, vocab):
        """Saves fasttext monolingual vectors in numpy array

        Args:
            vocab: dictionary vocab[word] = index

        """
        embeddings = np.zeros([len(vocab), self.dim])
        with open(self.emb_filename) as f:
            next(f)
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)

        np.savez_compressed(self.trimmed_filename, embeddings=embeddings)

    def export_trimmed_glove_vectors(self, vocab):
        """Saves glove vectors in numpy array

        Args:
            vocab: dictionary vocab[word] = index

        """
        embeddings = np.zeros([len(vocab), self.dim])
        with open(self.emb_filename) as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)

        np.savez_compressed(self.trimmed_filename, embeddings=embeddings)

    def export_trimmed_multi_vectors(self, vocab):
        """Saves glove vectors in numpy array

        Args:
            vocab: dictionary vocab[word] = index

        """
        embeddings = np.zeros([len(vocab), self.dim])
        with open(self.emb_filename) as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0].split("_")[1]
                embedding = [float(x) for x in line[1:]]
                if word in vocab:
                    word_idx = vocab[word]
                    embeddings[word_idx] = np.asarray(embedding)

        np.savez_compressed(self.trimmed_filename, embeddings=embeddings)

    def get_trimmed_vectors(self):
        """
        Args:
            filename: path to the npz file

        Returns:
            matrix of embeddings (np array)
        """
        try:
            with np.load(self.trimmed_filename) as data:
                print(data["embeddings"])
                return data["embeddings"]

        except IOError:
            raise Exception("Could not find or load file!!", self.trimmed_filename)

    def get_emb_vocab(self):
        """Load vocab from file

        Returns:
            vocab: set() of strings
        """
        print("Building vocab...")
        vocab = set()
        with open(self.emb_filename) as f:
            lines = f.readlines()
            for line in lines[1:]:
                word = line.strip().split(' ')[0]
                vocab.add(word)
        print("- done. {} tokens".format(len(vocab)))

        return vocab

    def get_multi_emb_vocab(self):
        """Load vocab from file

        Returns:
            vocab: set() of strings
        """
        print("Building vocab...")
        vocab = set()
        with open(self.emb_filename) as f:
            lines = f.readlines()
            for line in lines:
                word = line.strip().split(' ')[0].split("_")[1]
                vocab.add(word)
        print("- done. {} tokens".format(len(vocab)))

        return vocab

class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```

    """
    def __init__(self, filename, lang, mode, processing_word=None, processing_tag=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
        self.lang = lang
        self.mode = mode

    def __iter__(self):
        niter = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if len(line) == 0 or line.startswith("-DOCSTART-"):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags
                        words, tags = [], []
                else:
                    ls = line.split(' ')
                    word, tag = ls[0], ls[1]
                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    words += [word]
                    tags += [tag]

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(datasets):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for dataset in datasets:
        for words, _ in dataset:
            for word in words:
                vocab_char.update(word)

    return vocab_char


def get_processing_word(vocab_words=None, vocab_chars=None,
                        lowercase=False, chars=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        #print("len(vocab_words):", len(vocab_words))
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                    #print("len(vocab_words):", len(vocab_words))
                else:
                    raise Exception("Unknow key is not allowed. Check that " \
                                    "your vocab (tags?) is correct =>"+ str(len(vocab_words)))

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise Exception("Could not find or load file!!", filename)
    return d


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for lang in data:
        for (x, y) in data[lang]:
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []

            if type(x[0]) == tuple:
                x = zip(*x)
            x_batch += [x]
            y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def minibatches_test(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks


if __name__ == '__main__':

    args = get_args()

    if args.task == "trigger":
        task = "TriggerIdentification"
    elif args.task == "argument":
        task = "ArgumentIdentification"
    else:
        task = "Joint"

    ## Dataset Directory
    pre_dir = args.root_dir + args.pre_dir + "tagging-new/" + task + "/"

    iso_lang_dict, lang_iso_dict = get_iso_lang_abbreviation()

    processing_word = get_processing_word(lowercase=True)
    if args.use_neg_eg:
        ext = "_with_neg_eg"
    else:
        ext = "_wout_neg_eg"


    train = CoNLLDataset(pre_dir+"English/train" + ext + ".txt", "English", processing_word)



