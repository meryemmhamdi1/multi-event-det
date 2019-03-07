import glob
import get_args
from DataModule.data_preprocessor import *

if __name__ == '__main__':
    doc_events_dict = {}
    events_list_lang = {}

    args = get_args()

    root_path = args.root_dir + args.data_ace_path
    languages = [file_.split("/")[-1] for file_ in glob.glob(root_path + "*") if "Icon\r" not in file_]

    for language in languages:
        files_num = 0
        domains = [file_.split("/")[-1] for file_ in glob.glob(root_path + language + "/*" ) if "Icon\r" not in file_]
        events_lang = []
        for domain in domains:
            doc_events, num = extract_from_xml(root_path, language, domain)
            files_num += num
            for events_doc in doc_events:
                events_lang.extend(events_doc["events"])

        print("******** Language: "+language+ " Number of Processed files is: ", files_num)

        events_list_lang.update({language: events_lang})

