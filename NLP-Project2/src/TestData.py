'''
Created on Oct 15, 2015

@author: Mihir Shah -   mgs275 
         Hardik Patel - hvp4
'''

from xml.dom import minidom
from nltk import stem
import random
import re
import operator
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer
from XMLParser import senseidProbability, senseid
from nltk.stem.lancaster import LancasterStemmer



doc = minidom.parse("D:\\MEng folders\\NLP\\project2\\well formed xmls\\training-data.xml")
doc_test = minidom.parse("D:\\MEng folders\\NLP\\project2\\well formed xmls\\test-data.xml")
dictionary= minidom.parse("D:\\MEng folders\\NLP\\project2\\well formed xmls\\Dictionary.xml")

'''Test'''
lexelts_test = doc_test.getElementsByTagName("lexelt")
lexelts_dictionary=dictionary.getElementsByTagName("lexelt")
window_size_test=10           
trainingContextProbability={}
stemmer=stem.snowball.EnglishStemmer()
answers = doc.getElementsByTagName("answer")
lexelts = doc.getElementsByTagName("lexelt")
senseidProbability={}
senseIdCount={}
context_dictionary={}
window_size=20           
stop = stopwords.words('english')
trainingContextProbability={}
lexeltSenseIdMap={}

'''
The following loop parses the xml document and computes the following things
1. P(senseid) by using the formula count(si;w)/count(w)
2. Creating the context_dictionaries for respective sense_id which will have the senseid as a key and its value will be a list which will have the concatenation of 10 words from the left and 10 words from the right

'''

for lexelt in lexelts :
    #instances=doc.getElementsByTagName("instance")
    instances=lexelt.getElementsByTagName("instance")
    item_name = lexelt.getAttribute("item")
    lexeltSenseIdMap[item_name]=[]
    
    
    count=0; 
    senseid_probability_for_a_word={}           
    for instance in instances:
        answers=instance.getElementsByTagName("answer")
        contexts=instance.getElementsByTagName("context")[0]
        
        firstChild= contexts.firstChild.data
        lastChild= contexts.lastChild.data
        
        for c in string.punctuation:
            firstChild= firstChild.replace(c,"")
            lastChild=lastChild.replace(c,"")
        
        firstChild=firstChild.lower().split(" ")
        lastChild=lastChild.lower().split(" ")
        firstChild=list(filter(('').__ne__, firstChild))
        lastChild=list(filter(('').__ne__, lastChild))
        firstChild= [i for i in firstChild if i not in stop]
        lastChild= [i for i in lastChild if i not in stop]
        firstChild_context_elements= firstChild[-window_size:]
        lastChild_context_elements=lastChild[:window_size]
        
        firstChild_context_elements_stemmed=[LancasterStemmer().stem(word) for word in firstChild_context_elements]
        lastChild_context_elements_stemmed=[LancasterStemmer().stem(word) for word in lastChild_context_elements]

        #print(firstChild_context_elements_stemmed+ lastChild_context_elements_stemmed)
        #print(PorterStemmer().stem_word(contexts.data))
        
        for answer in answers:
            senseid=answer.getAttribute("senseid")
            
            temp_list=lexeltSenseIdMap[item_name]
            if senseid not in temp_list:
                lexeltSenseIdMap[item_name].append(senseid)
            
            
            
            if senseid in senseid_probability_for_a_word:
                senseid_probability_for_a_word[senseid]+=1
                senseIdCount[senseid]+=1
            else:                
                senseid_probability_for_a_word[senseid]=1
                senseIdCount[senseid]=1
            
            context_data=firstChild_context_elements_stemmed+lastChild_context_elements_stemmed
            #print(context_data)
            if senseid in context_dictionary:    
                context_dictionary[senseid]= context_dictionary[senseid]+ context_data
            else:
                context_dictionary[senseid]= firstChild_context_elements_stemmed+lastChild_context_elements_stemmed   
        count+=1
    senseid_probability_for_a_word.update((x, y/count) for x, y in senseid_probability_for_a_word.items())
    senseidProbability[item_name]=senseid_probability_for_a_word
    

    #print( "item name:%s, count: %s " %  (item_name, count))
#print(lexeltSenseIdMap)
#print(senseidProbability)
#print("context dictionary =")
#print(context_dictionary)
#print("-----------------------------")
#print(senseIdCount) 
print("Id,Prediction")


'''
The following loop does the following :
1. Converts the context dictionary in which value was a list into the dictionary of dictionary
  i.e the key is a particular sense_id and its value is another dictionary which will have the word as key and its count in the given context as a value

'''


for senseid in context_dictionary:
    temp_list=context_dictionary[senseid]
    temp_dictionary={} 
    for item in temp_list:
        if item in temp_dictionary:
            temp_dictionary[item]+=1
        else:
            temp_dictionary[item]=1
    context_dictionary[senseid]=temp_dictionary
    
    #temp_dictionary.update((x, y/senseIdCount[senseid]) for x, y in temp_dictionary.items())
    #trainingContextProbability[senseid]=temp_dictionary
    

for senseid in context_dictionary:
     temp_dictionary1={}
     temp_dictionary1=context_dictionary[senseid]
     count=senseIdCount[senseid]
     for word in temp_dictionary1:
         temp_dictionary1[word]= temp_dictionary1[word]/count
     trainingContextProbability[senseid]=temp_dictionary1

#print(context_dictionary)
#print('--------------')
#print(trainingContextProbability)



for lexelt in lexelts_test :
    #instances=doc.getElementsByTagName("instance")
    instances=lexelt.getElementsByTagName("instance")
    item_name = lexelt.getAttribute("item")
    count=0; 
    senseid_probability_for_a_word={}           
    for instance in instances:
        
        contexts=instance.getElementsByTagName("context")[0]
        
        firstChild= contexts.firstChild.data
        lastChild= contexts.lastChild.data
        
        for c in string.punctuation:
            firstChild= firstChild.replace(c,"")
            lastChild=lastChild.replace(c,"")
        
        firstChild=firstChild.lower().split(" ")
        lastChild=lastChild.lower().split(" ")
        firstChild=list(filter(('').__ne__, firstChild))
        lastChild=list(filter(('').__ne__, lastChild))
        firstChild= [i for i in firstChild if i not in stop]
        lastChild= [i for i in lastChild if i not in stop]
        firstChild_context_elements= firstChild[-window_size_test:]
        lastChild_context_elements=lastChild[:window_size_test]
        feature_vector= firstChild_context_elements+lastChild_context_elements
        feature_vector_stemmed=[LancasterStemmer().stem(word) for word in feature_vector]
        #print(feature_vector_stemmed)
        #print(PorterStemmer().stem_word(contexts.data))
        
        senseid_probability_predictions={}
        senseids=senseidProbability[item_name]
        senseidLabel=""
        maxSum=0.0
        for senseid in senseids:
            temp_sum=0.0
            
            for word in feature_vector:
                temp_dictionary=context_dictionary[senseid]
                if word in temp_dictionary:
                    temp_sum=temp_sum+temp_dictionary[word]
                    
            temp_sum=temp_sum*senseids[senseid]
            senseid_probability_predictions[senseid]=temp_sum
            
            if temp_sum>maxSum:
                maxSum=temp_sum
                senseidLabel=senseid
#             if senseidLabel == "" :
#                 senseidLabel=senseid
        
        if senseidLabel == "U":
         #print("inside blank")   
         senseidLabel=random.choice(list(senseids.keys()))
         #print("new="+ senseidLabel)                
        print(instance.getAttribute("id")+"\t"+senseidLabel)
        #print(senseidLabel)
        #print(PorterStemmer().stem_word(contexts.data))

    
               
        
        
    
   





    
       