import nltk

class NLTKLemmatizer:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def get_wordnet_pos(self, pos):
        """Map POS tag to first character lemmatize() accepts"""
        tag = pos[0].upper()
        tag_dict = {"J": "a",
                    "N": "n",
                    "V": "v",
                    "R": "r"}
        return tag_dict.get(tag, nltk.wordnet.NOUN)

    def lemmatize_sentence(self, sent):
        words = nltk.word_tokenize(sent)
        return [self.lemmatizer.lemmatize(w) for w in words]

    def lemmatize_paragraph(self, para):
        sentences = self.tokenizer.tokenize(para)
        for sent in sentences:
            res = self.lemmatize_sentence(sent)
            res_filtered = []
            for w in res:
                res_filtered.append(w)
            yield res_filtered

    def lemmatize_phrase(self, phrase):
        lemma = " ".join([self.lemmatizer.lemmatize(w) for w in phrase.split()])
        return lemma
