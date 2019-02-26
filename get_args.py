import argparse

def get_args():

    parser = argparse.ArgumentParser()

    ### Dataset Parameters
    parser.add_argument("--data-choice", "-dc", type=str,  default="ace05")
    parser.add_argument("--languages", "-lang", type=str,  default="English,Chinese,Arabic,Spanish")
    parser.add_argument("--root-dir", "-rd", type=str,  default="../../../3Datasets/")
    parser.add_argument("--data-ace-path", "-dp", type=str, default="ACE/Raw/ace_2005/data/")
    parser.add_argument("--pre-dir", "-pd", type=str, default="ACE/Preprocessed/")
    parser.add_argument("--w2v-dir", "-w2v", type=str, default="ace05")
    parser.add_argument("--train-mode", "-tm", type=str, default="ace05")

    parser.add_argument("--res-dir", "-pd", type=str, default="../../../6Results/")

    ### Model Parameters
    parser.add_argument("--dropout", "-dp", type=float, default=0.5)
    parser.add_argument("--learning-rate", "-lra", type=float, default=1e-3, help='')
    parser.add_argument("--beta-1", "-b1", type=float, default=0.7, help='')
    parser.add_argument("--beta-2", "-b2", type=float, default=0.999, help='')
    parser.add_argument("--epsilon", "-eps", type=float, default=1e-08, help='')

    return parser.parse_args()