import re
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class TextCleaning:
    def __init__(self, data):
        self.data = data
        self.data = str(data)

    def lowercasing(self):
        clean_text = self.data.lower()
        return clean_text

    def removing_stopwords(self):
        clean_text = remove_stopwords(self.data)
        return clean_text

    def removing_punctuation(self):
        clean_text = re.sub(r'[^\w\s]', '', self.data)
        return clean_text
    
    def removing_numbers(self):
        clean_text = re.sub(r'\d', '', self.data)
        return clean_text

    def removing_html_tags(self):
        re_html = re.compile(r'<.*?>')
        clean_text = re_html.sub('', self.data)
        return clean_text

