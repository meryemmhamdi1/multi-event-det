class ACEDocument(object):
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


class Event(object):
    def __init__(self, event_id, mention_id, type_, subtype, modality, polarity, genericity, tense, extent, extent_start, extent_end, scope, scope_start, scope_end, trig_text, trig_start, trig_end, arguments, entities):
        self.event_id = event_id
        self.mention_id = mention_id
        self.type_ = type_
        self.subtype = subtype
        self.modality = modality
        self.polarity = polarity
        self.genericity = genericity
        self.tense = tense
        self.extent = extent
        self.extent_start = extent_start
        self.extent_end = extent_end
        self.scope = scope
        self.scope_start = scope_start
        self.scope_end = scope_end
        self.trig_text = trig_text
        self.trig_start = trig_start
        self.trig_end = trig_end
        self.arguments = arguments
        self.entities = entities

    def to_string(self):
        return "Event: { event_id = " + self.event_id + "mention_id = " + self.mention_id + ", type = " + self.type_ + ", subtype = " +self.subtype + ", modality = " \
               + self.modality + ", polarity = " + self.polarity + ", genericity= " + self.genericity + ", tense = " + \
               self.tense + ", extent = " +self.extent + ", scope = " + self.scope  + ", trigger = " + self.trig_text

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class Trigger(object):
    def __init__(self, start, text, end, id_, event_type):
        self.start = start
        self.text = text
        self.end = end
        self.id_ = id_
        self.event_type = event_type

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class Argument(object):
    def __init__(self, id_, text, role, start, end, entity_type):
        self.id_ = id_
        self.text = text
        self.role = role
        self.start = start
        self.end = end
        self.entity_type = entity_type

    def to_string(self):
        return "Argument: {id_ = " + self.id_ + ", text = " + self.text + ", role = " + self.role + ", start =" + str(self.start) + ", end =" + str(self.end) + "}"

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class Entity(object):
    def __init__(self, id_, text, entity_type, phrase_type, start, end):
        self.id_ = id_
        self.text = text
        self.entity_type = entity_type
        self.phrase_type = phrase_type
        self.start = start
        self.end = end

    def to_string(self):
        return "Entity: {id_ = " + self.id_ + ", text = " + self.text + ", entity_type = " + self.entity_type + ", phrase_type=" + self.phrase_type + ", start =" + str(self.start) + ", end =" + str(self.end) + "}"

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)


class Sentence(object):
    def __init__(self, text, start, end):
        self.text = text
        self.start = start
        self.end = end

    def to_string(self):
        return "Sentence: {text = " + self.text + ", start = " + self.start + ", end = " + self.end + "}"

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self)
