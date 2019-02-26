class Event(object):
    def __init__(self, event_id, mention_id, type_, subtype, modality, polarity, genericity, tense, extent, scope, trigger, arguments):
        self.event_id = event_id
        self.mention_id = mention_id
        self.type_ = type_
        self.subtype = subtype
        self.modality = modality
        self.polarity = polarity
        self.genericity = genericity
        self.tense = tense
        self.extent = extent
        self.scope = scope
        self.trigger = trigger
        self.arguments = arguments


    def to_string(self):
        return "Event: { event_id = " + self.event_id + "mention_id = " + self.mention_id + ", type = " + self.type_ + ", subtype = " +self.subtype + ", modality = " \
               + self.modality + ", polarity = " + self.polarity + ", genericity= " + self.genericity + ", tense = " + \
               self.tense + ", extent = " +self.extent + ", scope = " + self.scope  + ", trigger = " + self.trigger
