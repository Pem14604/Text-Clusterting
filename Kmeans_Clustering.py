"""
Created on Thu May 16 10:39:05 2019

@author: vsolanki
"""


from gensim.models import Word2Vec
from nltk.cluster import KMeansClusterer
import nltk
import numpy as np 
from sklearn import cluster
from sklearn import metrics
import pandas as pd



# training data
  
paragraph = """Machine learning (ML) is the scientific study of algorithms and statistical
 models that computer systems use to effectively perform a specific task without using explicit
 instructions, relying on patterns and inference instead. It is seen as a subset of artificial
 intelligence. Machine learning algorithms build a mathematical model based on sample data, known
 as "training data", in order to make predictions or decisions without being explicitly programmed
 to perform the task.An earthquake (also known as a quake, tremor or temblor) is the shaking of the
 surface of the Earth, resulting from the sudden release of energy in the Earth's lithosphere that
 creates seismic waves. Earthquakes can range in size from those that are so weak that they cannot
 be felt to those violent enough to toss people around and destroy whole cities.ava is a general-purpose
 programming language that is class-based, object-oriented,[15] and designed to have as few implementation
 dependencies as possible. It is intended to let application developers "write once, run anywhere" (WORA),
 meaning that compiled Java code can run on all platforms that support Java without the need for recompilation.
 Java applications are typically compiled to "bytecode" that can run on any Java virtual machine (JVM) regardless of
 the underlying computer architecture.Compost decaying organic material used as a plant fertilizer."""
 
 
 
sent=nltk.sent_tokenize(paragraph)
sentences=[]
for sen in sent:
    x=sen.split()
    sentences.append(x)


#TFIDF#####################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 1, max_df =0.8, stop_words =('english'))
Tfidf= vectorizer.fit_transform(sent).toarray()
#Tfidf= vectorizer.fit_transform(sent) to find cosine similarity we dont need to convert matrix into array

cluster=4
model= KMeans(n_clusters=cluster, init='k-means++', max_iter=40, n_init=1)
model.fit(Tfidf)

# =============================================================================
# Tfidf= vectorizer.fit_transform(sent).toarray()
# NUM_CLUSTERS=4
# model = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', max_iter=40, n_init=1)
# model.fit(Tfidf)
# =============================================================================

order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

#showing top 10 words per cluster
for i in range(cluster):
    top_ten_words = [terms[ind] for ind in order_centroids[i, :10]]
    print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))    
    
    
        
x=int(len(terms)/cluster)
x=int(x)


#creating word and cluster dataframe
Tfidf_word_cluster={}
for i in range(cluster):
    print(i)
    #print("Cluster %d:" % i),
    for ind in order_centroids[i, :x]:
        print(terms[ind])
        #print(i,terms[ind])
        #print(' %s' % terms[ind]),
        Tfidf_word_cluster.setdefault(i,[]).append(terms[ind]) 

Tfidf_word_df=pd.DataFrame(Tfidf_word_cluster.items(), columns=['Cluster', 'word'])
Tfidf_word_df.head()
# =============================================================================
# Tfidf_word_df.to_csv(r'D:\Learning\Clustertng\Tfidf_word_df.csv')
# =============================================================================

#showing top 10 words per cluster
# =============================================================================
# for i in range(cluster):
#     top_ten_words = [terms[ind] for ind in order_centroids[i, :10]]
#     print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))    
#     
# print(model.labels_)    
# print(sent)
# =============================================================================

#creating sentence and cluster dataframe
Tfidf_sent_df= pd.DataFrame({
    'text': sent,
    'category': model.labels_
})
# =============================================================================
# Tfidf_sent_df.to_csv(r'D:\Learning\Clustertng\Tfidf_sent_df.csv')
# =============================================================================

# =============================================================================
# #predicting cluster of user query
# text_to_predict = "Machine learning is used to predict the future?"
# Y = vectorizer.transform([text_to_predict])
# predicted_cluster = model.predict(Y)
# 
# =============================================================================


#Word2vec_word #############################################################   


   
model = Word2Vec(sentences, min_count=1)
words = list(model.wv.vocab)
print (len(list(model.wv.vocab)))
word2vec_word= model[model.wv.vocab]


NUM_CLUSTERS=4
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(word2vec_word, assign_clusters=True)

word2vec_word_clsuter={} 
for index, sentence in enumerate(words):  
    word2vec_word_clsuter.setdefault(assigned_clusters[index],[]).append(sentence) 
    
word2vec_word_df=pd.DataFrame(word2vec_word_clsuter.items(), columns=['Cluster', 'word'])
word2vec_word_df.head()
# =============================================================================
# word2vec_word_df.to_csv(r'D:\Learning\Clustertng\word2vec_word_df.csv')  
# =============================================================================

    

#Word2vec_sent #####################################################################

model = Word2Vec(sentences, min_count=1)
  
def sent_vectorizer(sent, model):
    sent_vec =[]
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw+=1
        except:
            pass
     
    return np.asarray(sent_vec) / numw
  
  
Word2vec_sent=[]
for sentence in sentences:
    Word2vec_sent.append(sent_vectorizer(sentence, model))   
 
print (model[model.wv.vocab])
 
 
#print (model.similarity('post', 'book'))
#print (model.most_similar(positive=['machine'], negative=[], topn=2))
  
NUM_CLUSTERS=4
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(Word2vec_sent, assign_clusters=True)
print (assigned_clusters)
  
word2vec_sent_df = pd.DataFrame({
    'text': sent,
    'category': assigned_clusters
})

# =============================================================================
# word2vec_sent_df.to_csv(r'D:\Learning\Clustertng\word2vec_sent_df .csv')  
# =============================================================================
#for index, sentence in enumerate(sent):    
   # print (str(assigned_clusters[index]) + ":" + str(sentence))
   
   
# =============================================================================
# Kmeans_word2vec_sent={}
# for index, sentence in enumerate(sent):  
#     Kmeans_word2vec_sent.setdefault(assigned_clusters[index],[]).append(str(sentence)) 
# #print (str(assigned_clusters[index]) + ":" + str(sentence))    
# 
# print(Kmeans_Word2vec_sent)   
# =============================================================================



#Fasttext##################################################


from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText

feature_size = 50   # Word embedding vector dimensionality  
window_context = 2  # Context window size                                                                                    
min_word_count = 1  # Minimum word count                        
sample = 1e-3  


fasttext_model = FastText(sentences,
                          size=feature_size,
                          window=window_context,
                          min_count=min_word_count,
                          sample=sample,
                          sg=1, # sg decides whether to use the skip-gram model (1) or CBOW (0)
                          iter=50)

words = list(fasttext_model.wv.vocab)



#Fasttext Word ##################################################

fasttext_word_embb= fasttext_model[fasttext_model.wv.vocab]
cluster=4
kclusterer = KMeansClusterer(cluster, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(fasttext_word_embb, assign_clusters=True)

fasttext_word_cluster={} 
for index, sentence in enumerate(words):  
    fasttext_word_cluster.setdefault(assigned_clusters[index],[]).append(sentence) 

fasttext_word_df=pd.DataFrame(fasttext_word_cluster.items(), columns=['Cluster', 'word'])
# =============================================================================
# fasttext_word_df.head()
# fasttext_word_df.to_csv(r'D:\Learning\Clustertng\fasttext_word_df.csv')  
# 
# 
# =============================================================================
#Fasttext Sent ##################################################


fasttext_sent_embb=[]
for sentence in sentences:
    fasttext_sent_embb.append(sent_vectorizer(sentence, fasttext_model)) 
    
cluster=4
kclusterer = KMeansClusterer(cluster, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(fasttext_sent_embb, assign_clusters=True)

fasttext_sent_df= pd.DataFrame({
    'text': sent,
    'category': assigned_clusters
})
# =============================================================================
# fasttext_sent_df.to_csv(r'D:\Learning\Clustertng\fasttext_sent_df.csv') 
# =============================================================================


# =============================================================================
# 
# # cosinle similarity, for this we need data in matrix form and to conver the fasttext embeddings in matrix as TFIDF matrix
# we need to covert the embeddings intoarray and then in matrix
# =============================================================================

# =============================================================================
# import numpy as np
# fasttext_sent_embb_array= np.asarray(fasttext_sent_embb)
# b = np.matrix(fasttext_sent_embb_array)
# 
# from sklearn.metrics.pairwise import cosine_similarity
# sim=cosine_similarity(b[1], b)
# array([[ 1.        ,  0.36651513,  0.52305744,  0.13448867]])
# 
# =============================================================================



