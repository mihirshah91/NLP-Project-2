import nltk
from nltk.corpus import brown
from nltk import PorterStemmer
list_words=['class', 'laity', 'continue', 'important', 'role', 'laity', 'judged','judgemental','judgement','judges', 'religious','religion', 'goals', 'personally']

#list_stemmed_words=[PorterStemmer().stem_word(word for word in  list_words)]
for word in list_words:
    print(PorterStemmer().stem_word(word))


#print(PorterStemmer().stem_word( (word) for word in  list_words))
print(brown.words())

'''text = nltk.word_tokenize("And now for something completely different")'''
#print(nltk.pos_tag("and"))