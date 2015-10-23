'''
Created on Oct 15, 2015

@author: Mihir Shah -   mgs275 
         Hardik Patel - hvp4
'''

from xml.dom import minidom
from nltk import stem
import string
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import wordnet as wn
import csv
import math



doc = minidom.parse("D:\\MEng folders\\NLP\\project2\\well formed xmls\\training-data.xml")
doc_test = minidom.parse("D:\\MEng folders\\NLP\\project2\\well formed xmls\\test-data.xml")
dictionary= minidom.parse("D:\\MEng folders\\NLP\\project2\\well formed xmls\\Dictionary.xml")

'''Test'''
lexelts_test = doc_test.getElementsByTagName("lexelt")
lexelts_dictionary=dictionary.getElementsByTagName("lexelt")
window_size_test=15     

answers = doc.getElementsByTagName("answer")
lexelts = doc.getElementsByTagName("lexelt")
senseidProbability={}
senseIdCount={}
context_dictionary={}
window_size=15           
stop = stopwords.words('english')


'''
The following loop parses the xml document and computes the following things
1. P(senseid) by using the formula count(si;w)/count(w)
2. Creating the context_dictionaries for respective sense_id which will have the senseid as a key and its value will be a list which will have the concatenation of 10 words from the left and 10 words from the right

'''

for lexelt in lexelts :
    #instances=doc.getElementsByTagName("instance")
    instances=lexelt.getElementsByTagName("instance")
    item_name = lexelt.getAttribute("item")
    
    
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
        firstChild_context_elements= firstChild
        lastChild_context_elements=lastChild
        
        firstChild_context_elements_stemmed=[LancasterStemmer().stem(word) for word in firstChild_context_elements]
        lastChild_context_elements_stemmed=[LancasterStemmer().stem(word) for word in lastChild_context_elements]

        #print(firstChild_context_elements_stemmed+ lastChild_context_elements_stemmed)
        #print(PorterStemmer().stem_word(contexts.data))
        
        for answer in answers:
            senseid=answer.getAttribute("senseid")
            if senseid == 'U':
                senseid= senseid+item_name
              
                  
            
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
#print("senseid probability=")
#print(senseidProbability)

f = open("D:\\MEng folders\\NLP\\project2\\senseidProb.txt",'w')

f.write(str(senseidProbability))
f.close()
#print("senseid count=")
f = open('D:\\MEng folders\\NLP\\project2\\senseIdCount.txt','w')
f.write(str(senseIdCount))
f.close()
#print(senseIdCount)
#print("context dictionary =")
f = open('D:\\MEng folders\\NLP\\project2\\context_dictionaryRaw.txt','w')
f.write(str(context_dictionary))
f.close()
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


f = open('D:\\MEng folders\\NLP\\project2\\context_dictionaryCount.txt','w')
f.write(str(context_dictionary)) 
f.close()    

for senseid in context_dictionary:
     temp_dictionary1={}
     temp_dictionary1=context_dictionary[senseid]
     count=senseIdCount[senseid]
     for word in temp_dictionary1:
         temp_dictionary1[word]= temp_dictionary1[word]/count
     

#print(context_dictionary)
#print('--------------')
#print(trainingContextProbability)

f = open('D:\\MEng folders\\NLP\\project2\\context_dictionaryProb.txt','w')
f.write(str(context_dictionary)) 
f.close()    

print("done outputting to a file")

f = open('D:\\MEng folders\\NLP\\project2\\Prediction321.csv','w')
fcsv = csv.writer(f, delimiter=',')
fcsv.writerows("Id,Prediction")

f2 = open('D:\\MEng folders\\NLP\\project2\\temp_sums.txt','w')
f3 = open('D:\\MEng folders\\NLP\\project2\\featureVectors.txt','w')



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
        firstChild_context_elements= firstChild
        lastChild_context_elements=lastChild
        feature_vector= firstChild_context_elements+lastChild_context_elements
        
        featureVectorSynset=[]
        for word in feature_vector:
            synsets= wn.synsets(word)
            list_syn=[]

            for ss in synsets:
                temp=ss.name()
                temp=temp.split(".")[0]
                list_syn.append(temp)
            syn_set=set(list_syn)
            syn_set=list(syn_set) 
            syn_set=syn_set[:2] 
            featureVectorSynset=featureVectorSynset+ syn_set
        #print(featureVectorSynset)
        feature_vector= list(set(feature_vector+featureVectorSynset))
        #print(feature_vector)
        feature_vector_stemmed=[LancasterStemmer().stem(word) for word in feature_vector]
        
        
        
        feature_vector_stemmed_dict = dict()
        for i in feature_vector_stemmed:
            feature_vector_stemmed_dict[i] = feature_vector_stemmed_dict.get(i, 0) + 1
        #print(feature_vector_stemmed)
        #print(PorterStemmer().stem_word(contexts.data))
        f3.write(','.join(feature_vector_stemmed)+ "for instanceid "+ str(instance.getAttribute("id")) + "\n")
        
        
        senseids=senseidProbability[item_name]
        senseidLabel=""
        maxSum=0.0
        for senseid in senseids:
            
                temp_sum=1.0
                temp_dictionary=context_dictionary[senseid]
                for word in feature_vector_stemmed_dict:
                    
                    if word in temp_dictionary:
                        temp_sum=temp_sum * (temp_dictionary[word]**feature_vector_stemmed_dict[word]) # probability raised to count of that word
                    
                temp_sum=temp_sum *(senseids[senseid])
                
                f2.write(str(temp_sum) + " " + str(senseid) + " " )
                               
                #print(temp_sum," ", senseid," ",end='')
                
                if senseid == "U" + item_name:
                    senseid="U"
                
                if maxSum == 0.0:
                    maxSum= temp_sum
                    senseidLabel=senseid
                else:
                    if temp_sum-maxSum < 0.00001 and temp_sum-maxSum > -0.00001 :
                        maxSum=temp_sum
                        senseidLabel=senseidLabel+ " "+ senseid
                    else:
                         if temp_sum > maxSum:
                             maxSum=temp_sum
                             senseidLabel=senseid

         
        
        f2.write("for instance" + str(instance.getAttribute("id")))
        f2.write("\n")
        fcsv.writerows(str(instance.getAttribute("id"))+ "," +str(senseidLabel)) 
            

                         
        print(instance.getAttribute("id")+"\t"+senseidLabel)
f.close()
f2.close()
f3.close()
        #print(senseidLabel)
        #print(PorterStemmer().stem_word(contexts.data))

    
               
        
        
    
   





    
       