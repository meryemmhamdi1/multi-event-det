class Document(object):
    def __init__(self, id_, source, datetime, text):
        self.id_ = id_
        self.source = source
        self.datetime = datetime
        self.text = text

    def to_string(self):
        text_str = "["
        for t in self.text:
            text_str += t + ","
        text_str += "]"
        return "Document: {id_ = " + self.id_ + ", source = " + self.source + ", datetime = " + self.datetime  + ", text = " + text_str + "}"
