import pymorphy2
import nltk
import pandas as pd
import stanza
import spacy_stanza
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


stanza.download("ru")
nltk.download('punkt')
nlp = spacy_stanza.load_pipeline("ru")
nltk.download('stopwords')


stopword = stopwords.words('russian')
morph = pymorphy2.MorphAnalyzer()
tokenizer = RegexpTokenizer(r'\w+')


df = pd.read_csv("data_unprocessed.csv", delimiter=';')
df['sentiment'] = pd.to_numeric(df['sentiment'])


def preprocessing(texts):
    preprocessed_texts = []
    for text in texts:
        text_stripped = text.rstrip()
        text_stripped = ' '.join(tokenizer.tokenize(text_stripped.lower()))
        lemmas = [morph.parse(word)[0].normal_form for word in text_stripped.split(' ')]
        lemmas = [w for w in lemmas if not w.isdigit() and w != ' ' and w not in stopword]
        preprocessed_texts.append(' '.join(lemmas))

    return preprocessed_texts




def get_morph_tags(texts):
    pos_tags = []
    for text in texts:
        doc = nlp(text)
        text_pos_tags = [token.pos_ for token in doc]
        text_pos_tags = ['None' if v is None else v for v in text_pos_tags]
        text_pos_tags = [pos for pos in text_pos_tags if pos != 'PUNCT']
        text_pos_tags = ' '.join(text_pos_tags)
        pos_tags.append(text_pos_tags)

    return pos_tags


if __name__ == '__main__':
    df['preprocessed'] = preprocessing(df['content'])
    df['pos_tags'] = get_morph_tags(df['content'])
    df.to_csv('data.csv')


