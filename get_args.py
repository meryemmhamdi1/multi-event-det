import argparse


def get_args():

    parser = argparse.ArgumentParser()

    ### Dataset Parameters
    parser.add_argument("--languages", "-lang", type=str,  default="English,Chinese,Arabic")
    #parser.add_argument("--root-dir", "-rd", type=str,  default="../../../3Datasets/")
    parser.add_argument("--root-dir", "-rd", type=str,  default="../Datasets/")
    parser.add_argument("--data-ace-path", "-dap", type=str,
                        default="EventsExtraction/ACE/Raw/ACE2005-TrainingData-V6.0/")
    #parser.add_argument("--pre-dir", "-pd", type=str, default="EventsExtraction/ACE/Preprocessed/tagging-new/")
    parser.add_argument("--pre-dir", "-pd", type=str, default="ACE05_new_split_wout_neg/")
    parser.add_argument("--w2v-dir", "-w2v", type=str, default="")
    parser.add_argument("--method", "-mth", type=str, default="tagging", help="jmee or tagging")
    parser.add_argument("--train-mode", "-tm", type=str, default="ace05")
    parser.add_argument("--doc-splits", "-dspl", type=str, default="doc_splits/")
    parser.add_argument("--use-neg-eg", "-une", type=bool, default=False)
    parser.add_argument("--jmee-splits", "-js", type=str, default="doc_splits/English/jmee_split/")
    parser.add_argument("--split-option", "-so", type=str, default="randoc",
                        help="- cv for cross-validation"
                             "- jmee for jmee-splits"
                             "- randoc for another doc split"
                             "- ransent for another sent split")

    parser.add_argument("--train-prop", "-tr-pr", type=float, default=0.88)
    parser.add_argument("--test-prop", "-te-pr", type=float, default=0.07)
    parser.add_argument("--dev-prop", "-de-pr", type=float, default=0.05)

    ### Embeddings Parameters
    parser.add_argument("--dim-word", "-dw", type=int, default=300)
    parser.add_argument("--dim-char", "-dc", type=int, default=100)
    parser.add_argument("--use-pretrained", "-upre", type=bool, default=True)
    parser.add_argument("--train-embed", "-temb", type=bool, default=True)
    parser.add_argument("--embed-choice", "-embc", type=str, default="fasttext",
                        help="Family of embeddings to be used for monolingual and multilingual experiments in parallel"
                             "bert, fasttext, glove, muse, pseudo_dict, expert_dict etc ")
    #parser.add_argument("--embed-mono-path", "-emomp", type=str, default="../../../4Embeddings/Monolingual/")
    parser.add_argument("--embed-mono-path", "-emomp", type=str, default="../Embeddings/MonolingualEmbeddings/")
    #parser.add_argument("--embed-multi-path", "-emup", type=str, default="../../../4Embeddings/Multilingual/")
    parser.add_argument("--embed-multi-path", "-emup", type=str, default="../Embeddings/MultilingualEmbeddings/")

    ### Model Parameters
    parser.add_argument("--epoch", "-ep", type=int, default=25)
    parser.add_argument("--dropout", "-dp", type=float, default=0.5)
    parser.add_argument("--batch-size", "-bs", type=int, default=20)
    parser.add_argument("--optimizer", "-opt", type=str, default="adam")
    parser.add_argument("--learning-rate", "-lra", type=float, default=1e-1)
    parser.add_argument("--lr-decay", "-lrd", type=float, default=0.9)
    parser.add_argument("--clip", "-cl", type=int, default=-1)
    parser.add_argument("--n-epoch-no-imp", "-earep", type=int, default=10)
    parser.add_argument("--beta-1", "-b1", type=float, default=0.7)
    parser.add_argument("--beta-2", "-b2", type=float, default=0.999)
    parser.add_argument("--epsilon", "-eps", type=float, default=1e-08)
    parser.add_argument("--hidden-size-char", "-hsc", type=int, default=100)
    parser.add_argument("--hidden-size-lstm", "-hsl", type=int, default=300)
    parser.add_argument("--use-crf", "-ucr", type=bool, default=True)
    parser.add_argument("--use-chars", "-uch", type=bool, default=True)

    ### Experiment Parameters
    parser.add_argument("--mode", "-m", type=str, default="mono",
                        help="Modes of training the embeddings:"
                             "-- mono for monolingual"
                             "-- multi for training using multilingual embeddings")

    parser.add_argument("--model-choice", "-mc", type=str, default="biLSTMCRFChar",
                        help="Choice of the model to be used for training the tagging:"
                             "-- biLSTMCRFChar: Bidirectional LSTM on top of character and word embeddings CRF layer at the end"
                             "-- transformer ")

    parser.add_argument("--task", "-t", type=str, default="trigger",
                        help="Available options: "
                             "- trigger for Trigger Identification only,"
                             "- argument for ArgumentIdentification only"
                             "- joint (for both tasks)")

    parser.add_argument("--train-lang", "-trl", type=str, default="English",
                        help="The language on which the model is trained , it can be only one (monolingual) "
                             "or multilingual using multiple languages (separated by comma)")

    parser.add_argument("--test-lang", "-tel", type=str, default="English",
                        help="The language on which the model is tested , it can be only one (monolingual) "
                             "or multilingual using multiple languages (separated by comma)")


    ### Saving Results Option
    #parser.add_argument("--dir-output", "-do", type=str, default="../../../6Results/SequenceTagging/")
    parser.add_argument("--dir-output", "-do", type=str, default="../Results/SequenceTagging/")

    return parser.parse_args()
