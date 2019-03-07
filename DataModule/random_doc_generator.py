import glob
from get_args import *
import random as rd
import math

SEED = 1245789709644757090796

if __name__ == '__main__':

    args = get_args()
    languages = args.languages.split(",")
    inp_data_path = args.root_dir + args.data_ace_path

    splits_prop = {"train": args.train_prop, "test": args.test_prop, "dev": args.dev_prop}

    files_dict_lang = {}

    for lang in languages:
        print("Processing Language: ", lang)

        # Concatenating all documents in all domains
        if lang == "English":
            anno_vers = "timex2norm"
        else:
            anno_vers = "adj"
        for domain in [file_.split("/")[-1] for file_ in glob.glob(inp_data_path + lang + "/*") if "Icon\r" not in file_]:
            for file in glob.glob(inp_data_path + lang + "/" + domain + "/"+ anno_vers + "/*.apf.xml"):
                file_path = ".".join("/".join(file.split("/")[-3:]).split(".")[:-2])
                if lang not in files_dict_lang:
                    files_dict_lang.update({lang: [file_path]})
                else:
                    files_dict_lang[lang].append(file_path)

        print("Total number of Documents in >>>>", len(files_dict_lang[lang]))

        print("Splitting into train, test and dev")

        rd.seed(SEED)
        rd.shuffle(files_dict_lang[lang])

        start = 0
        for split in splits_prop:
            with open("doc_splits/" + lang + "/" + split, "w") as file:
                split_num = math.floor(splits_prop[split] * len(files_dict_lang[lang]))
                if split == "dev":
                    end = len(files_dict_lang[lang])
                else:
                    end = start+split_num

                print("Number of ", split, " is >>", end-start)
                for doc_path in files_dict_lang[lang][start:end]:
                    file.write(doc_path + "\n")
                start = start + split_num

        print("Done!")

