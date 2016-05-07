import re
from nltk.stem.snowball import SnowballStemmer
#SnowBall stemmer
stemmer = SnowballStemmer("english")


def clean_text(str):
    #Cleans the string from characters (only alphanumeric remains)
    #Does stemming
    str = re.sub('[^0-9a-zA-Z]+', ' ', str)
    str = str.lower()
    return ' '.join(map(stemmer.stem, str.split()))

def count_words(search_term,text):
    #Counts how many times search term appears in the text
    return sum([text.split().count(word) for word in search_term.split()])

