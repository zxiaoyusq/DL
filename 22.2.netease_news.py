

import numpy as np
from gensim import corpora, models, similarities
from pprint import pprint
import time

def chinese_word_cut(text):
    return " ".join(jieba.cut(text))

def load_stopword():                                                                                                                                                                                                                                                                   
    f_stop = open('stopword.txt')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw


if __name__ == '__main__':
    t_start = time.time()
    stop_words = load_stopword()
    
    inputfile='datascience.csv'
    data=pd.read_csv(inputfile,encoding='gb18030')
    data['content_cuted']=data['content'].apply(chinese_word_cut)  
    data['content_cuted'] = [[word for word in line.strip().lower().split() if word not in stop_words] for line in data['content_cuted']]
    
    M = len(data['content_cuted'])
    print '文本数目：%d个' % M

    dictionary = corpora.Dictionary(data['content_cuted'])
    V = len(dictionary)
    print('词的个数：', V)
    corpus = [dictionary.doc2bow(text) for text in data['content_cuted']]
    t_start = time.time()
    corpus_tfidf = models.TfidfModel(corpus)[corpus]

    num_topics = 5
    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                            alpha=0.01, eta=0.01, minimum_probability=0.001,
                            chunksize = 100, passes = 1)

    num_show_topic = 5  # 每个文档显示前几个主题
    print '10个文档的主题分布：'
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    idx = np.arange(M)
    np.random.shuffle(idx)
    idx = idx[:10]
    for i in idx:
        print(doc_topics[i])
        print(lda.show_topic(i,num_show_topic))

