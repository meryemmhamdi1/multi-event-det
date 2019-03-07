import argparse


def get_args():

    parser = argparse.ArgumentParser()

    ### Dataset Parameters
    parser.add_argument("--languages", "-lang", type=str,  default="English,Chinese,Arabic")
    parser.add_argument("--root-dir", "-rd", type=str,  default="../../../../3Datasets/")
    parser.add_argument("--data-ace-path", "-dap", type=str,
                        default="EventsExtraction/ACE/Raw/ACE2005-TrainingData-V6.0/")
    parser.add_argument("--pre-dir", "-pd", type=str,
                        default="EventsExtraction/ACE/Preprocessed/")
    parser.add_argument("--w2v-dir", "-w2v", type=str, default="")
    parser.add_argument("--res-dir", "-res", type=str, default="../../../6Results/")
    parser.add_argument("--task", "-tk", type=str, default="tagging", help="jmee or tagging")
    parser.add_argument("--train-mode", "-tm", type=str, default="ace05")
    parser.add_argument("--doc-splits", "-dspl", type=str, default="doc_splits/")
    parser.add_argument("--jmee-splits", "-js", type=str, default="doc_splits/English/jmee_split/")
    parser.add_argument("--split-option", "-so", type=str, default="jmee",
                        help="- cv for cross-validation"
                             "- jmee for jmee-splits"
                             "- randoc for another doc split"
                             "- ransent for another sent split")

    parser.add_argument("--train-prop", "-tr-pr", type=float, default=0.88)
    parser.add_argument("--test-prop", "-te-pr", type=float, default=0.07)
    parser.add_argument("--dev-prop", "-de-pr", type=float, default=0.05)

    ### Model Parameters
    parser.add_argument("--dropout", "-dp", type=float, default=0.5)
    parser.add_argument("--learning-rate", "-lra", type=float, default=1e-3, help='')
    parser.add_argument("--beta-1", "-b1", type=float, default=0.7, help='')
    parser.add_argument("--beta-2", "-b2", type=float, default=0.999, help='')
    parser.add_argument("--epsilon", "-eps", type=float, default=1e-08, help='')

    return parser.parse_args()
