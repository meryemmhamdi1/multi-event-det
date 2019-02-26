class Argument(object):
    def __init__(self, id_, role, text):
        self.id_ = id_
        self.role = role
        self.text = text

    def to_string(self):
        return "Argument: {id_ = " + self.id_ + ", role = " + self.role + ", text = " + self.text + "}"

