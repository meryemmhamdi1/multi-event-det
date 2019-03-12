import glob
import get_args
from DataModule.data_utils import *
from DataModule.data_preprocessor import *
import pickle as pkl
from EventDetectionModels.NNModels.bi_lstm_crf_char_emb_model import *

if __name__ == '__main__':
    args = get_args()

    ## Dataset Directory
    if args.task == "trigger":
        task = "TriggerIdentification"
    elif args.task == "argument":
        task = "ArgumentIdentification"
    else:
        task = "Joint"

    pre_dir = args.root_dir + args.pre_dir + task + "/"

    iso_lang_dict, lang_iso_dict = get_iso_lang_abbreviation()

    processing_word = get_processing_word(lowercase=True)
    if args.use_neg_eg:
        ext = "_with_neg_eg"
    else:
        ext = "_wout_neg_eg"

    if args.mode == "mono":
        test_languages = [args.test_lang]
        train_languages = [args.train_lang]
        res_dir = args.dir_output + args.model_choice + "test_doc_split" + "/" + args.test_lang + "/" + task + "/" + ext + "/"
    else:
        test_languages = args.test_lang.split(",")
        train_languages = args.train_lang.split(",")
        res_dir = args.dir_output + args.model_choice + "test_doc_split" + "/" + "".join(train_languages) \
                  + "TO" + "".join(test_languages) + "/" + task + "/" + ext + "/"
    print("res_dir:", res_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    args.res_dir = res_dir
    args.dir_model = res_dir + "model.weights/"

    ## Building Train/Dev/Test depending on train and test languages
    train = {}
    dev = {}
    test = {}
    for lang in train_languages:
        train.update({lang: CoNLLDataset(pre_dir+lang+"/train" + ext + ".txt", lang, args.mode, processing_word)})
        dev.update({lang: CoNLLDataset(pre_dir+lang+"/dev" + ext + ".txt", lang, args.mode, processing_word)})

    for lang in test_languages:
        test.update({lang: CoNLLDataset(pre_dir+lang+"/test" + ext + ".txt", lang, processing_word)})


    ## Create/Load Word,Tags and Chars Vocabulary
    words_vocab_file = res_dir+"words.txt"
    char_vocab_file = res_dir+"char.txt"
    tags_vocab_file = res_dir+"tags.txt"

    if not os.path.isfile(words_vocab_file) or not os.path.isfile(char_vocab_file) or not os.path.isfile(tags_vocab_file):
        datasets = []
        for lang in train:
            datasets.append(train[lang])
            datasets.append(dev[lang])

        for lang in test:
            datasets.append(test[lang])

        vocab_words, vocab_tags = get_vocabs(datasets)
        vocab_chars = get_char_vocab(datasets)
        write_vocab(vocab_words, words_vocab_file)
        write_vocab(vocab_chars, char_vocab_file)
        write_vocab(vocab_tags, tags_vocab_file)
    else:
        print("Loading from files", words_vocab_file)
        vocab_words = load_vocab(words_vocab_file)
        vocab_chars = load_vocab(char_vocab_file)
        vocab_tags = load_vocab(tags_vocab_file)

    # get pre-trained embeddings
    print("Get pre-trained embeddings >>>>>")
    if args.mode == "mono":
        if args.embed_choice == "fasttext":
            embed_filename = args.embed_mono_path + "wiki." + lang_iso_dict[args.train_lang] + ".vec"
    else:
        if args.embed_choice == "pseudo_dict" or args.embed_choice == "expert_dict":
            embed_filename = args.embed_multi_path + "expert_dict_dim_red_en_ar_zh.txt"

    emb = Embeddings(embed_filename,  args.embed_choice, vocab_words, args.dim_word)
    args.embeddings = (emb.get_trimmed_vectors() if args.use_pretrained else None)

    words = vocab_words.keys() & emb.embed_vocab
    words.add(UNK)
    words.add(NUM)
    args.vocab_words = {}
    for i, word in enumerate(words):
        args.vocab_words[word] = i

    print("args.vocab_words[UNK]:", args.vocab_words[UNK])
    print("len(args.vocab_words):", len(args.vocab_words))
    args.vocab_tags = vocab_tags
    args.vocab_chars = vocab_chars

    args.nwords = len(args.vocab_words)
    args.nchars = len(vocab_chars)
    args.ntags = len(vocab_tags)

    # 2. get processing functions that map str -> id
    args.processing_word = get_processing_word(args.vocab_words, vocab_chars, lowercase=True, chars=args.use_chars)
    args.processing_tag = get_processing_word(vocab_tags, lowercase=False, allow_unk=True)

    train = {}
    dev = {}
    test = {}
    for lang in train_languages:
        train.update({lang: CoNLLDataset(pre_dir+lang+"/train" + ext + ".txt", lang, args.mode, args.processing_word, args.processing_tag)})
        dev.update({lang: CoNLLDataset(pre_dir+lang+"/dev" + ext + ".txt", lang, args.mode, args.processing_word, args.processing_tag)})

    for lang in test_languages:
        test.update({lang: CoNLLDataset(pre_dir+lang+"/test" + ext + ".txt", lang, args.mode, args.processing_word, args.processing_tag)})

    ## Build Model
    args.log_filename = res_dir + "log.txt"
    model = BiLSTMCRFChar(args)
    metrics_train, metrics_dev, metrics_test, losses = model.train(train, dev, test)

    ## Saving results
    results_dict = {"train": metrics_train, "dev": metrics_dev, "train_loss": losses}
    #for lang in metrics_test:
    results_dict.update({"test": metrics_test})

    with open(res_dir + "dimword-"+str(args.dim_word)+"_dimchar-"+str(args.dim_char)+"_embedchoice-"+args.embed_choice+"_"+"_results.p", "wb") as dict_pkl:
        pkl.dump(results_dict, dict_pkl)










