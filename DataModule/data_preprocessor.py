import glob
import xml.etree.ElementTree as ET
import re
from document import *
from event import *
from argument import *

class DataPreprocessor(object):
    def __init__(self, data_util, mode):
        self.data_util = data_util
        self.mode = mode

    def extract_doc_info(self, root):
        for docid in root.iter('DOCID'):
            doc_id = docid.text

        for doctype in root.iter('DOCTYPE'):
            source = doctype.attrib['SOURCE']

        for datetime in root.iter('DATETIME'):
            datetime = datetime.text

        """
        for body in root.iter('BODY'):
            for headline in body.iter('HEADLINE'):
                headline = headline.text
                
        """

        turns = []
        for _ in root.iter('TURN'):
            for sp in root.iter('SPEAKER'):
                turns.append(sp.tail)

        return Document(doc_id, source, datetime, turns)

    def extract_event_info(self, event):
        event_id = event.attrib["ID"]
        event_type = event.attrib["TYPE"]
        subtype = event.attrib["SUBTYPE"]
        modality = event.attrib["MODALITY"]
        polarity = event.attrib["POLARITY"]
        genericity = event.attrib["GENERICITY"]
        tense = event.attrib["TENSE"]

        for mention in event.iter('event_mention'):
            mention_id = mention.attrib["ID"]
            for child in mention:
                if child.tag == "extent":
                    for chil2 in child:
                        extent = chil2.text
                elif child.tag == "ldc_scope":
                    for chil2 in child:
                        scope = chil2.text
                elif child.tag == "anchor":
                    for chil2 in child:
                        trigger = chil2.text

            arguments = []
            for argument in mention.iter('event_mention_argument'):
                arg_id = argument.attrib["REFID"]
                role = argument.attrib["ROLE"]
                for child in argument:
                    for chil2 in child:
                        arg_text = chil2.text
                arg = Argument(arg_id, role, arg_text)

                arguments.append(arg)

            ev = Event(event_id, mention_id, event_type, subtype, modality, polarity, genericity, tense, extent, scope, trigger, arguments)

        return ev

    def extract_from_xml(self, root_path, language, domain):
        doc_events = []
        #print(root_path + language + "/" + domain + "/adj/*.apf.xml")
        files_num = 0
        for file_name in sorted(glob.glob(root_path + language + "/" + domain + "/adj/*.apf.xml")):
            # Get the raw document
            raw_path = root_path + language + "/" + domain + "/adj/" + file_name.split("/")[-1].split(".apf.xml")[0] + ".sgm"
            print(raw_path)

            tree =  ET.parse(raw_path, ET.XMLParser(encoding='utf-8'))
            root = tree.getroot()

            doc = self.extract_doc_info(root)

            # Get the event + argument annotation
            tree = ET.parse(file_name, ET.XMLParser(encoding='utf-8'))
            root = tree.getroot()

            events = []
            for event in root.iter('event'):
                events.append(self.extract_event_info(event))

            doc_events.append({"doc": doc, "events": events})

            files_num += 1

        return doc_events, files_num