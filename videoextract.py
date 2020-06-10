from subprocess import call
import os
import cv2 as cv
import math
import glob
import websocket
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import hmac
import hashlib
import base64
from urllib.parse import urlencode
import ssl
import _thread as thread
import json
import time
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet
from pywsd.lesk import simple_lesk
import numpy as np
import requests
from PIL import Image
import networkx as nx
import re
import operator
import chardet

import summa


#求出一部视频总的时间
def sumTime(path):
    cap=cv.VideoCapture(path)
    s=cap.get(7)
    v=cap.get(5)
    t=s/v
    return math.ceil(t)

#将时间格式转换成00:00:00
def time_fomat(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    string = "%02d:%02d:%02d" % (h, m, s)
    return string

def stopword_list(stopwords_path):
    with open (stopwords_path,'r') as file:
        lines = file.readlines()
        stopwords_list = [line.strip() for line in lines]
        return stopwords_list

#格式为[a,b,c]
def remove_stopwords(wordlist,stopwords):
    result_words = []
    for word in wordlist:
        if word.lower() not in stopwords and word != ' ' and word != '\u3000':
            result_words.append(word)
    return result_words

def normalization(dist):
    sim = 0
    if math.isinf(dist):
        sim = 0
    else:
        sim = 1.0 / (1.0 + dist)
    return sim

#----------------------------计算标题相关性---------------------
#获得关键词语义列表（词语列表转成语义列表
def words_to_senses(wordslist,sentences):
    senseslist = []
    for i in range(len(wordslist)):
        words = wordslist[i]
        sentence = sentences[i]
        senses = []
        for w in words:
            sense = simple_lesk(sentence, w) #可指定pos='n'
            if not sense:
                sense = None
            else:
                senses.append(sense)
        senseslist.append(senses)
    return senseslist

#将标题词语列表转成语义列表
def title_words_to_senses(title_words,title):
    title_senses = []
    for tw in title_words:
        tw_sense = simple_lesk(title, tw)
        if not tw_sense:
            continue
        else:
            title_senses.append(tw_sense)
    return title_senses

#计算标题相关性(取最大值)
def title_sim(wordslist, senseslist, title_senses):
    title_map = {}
    for words in wordslist:
        for word in words:
            title_map[word]=0
    for i in range(len(senseslist)):
        senses = senseslist[i]
        for j in range(len(senses)):
            s = senses[j]
            if s:
                s_n = wordnet.synset(s.name())
                max_sim = 0
                for ts in title_senses:
                    ts_n = wordnet.synset(ts.name())
                    sim = s_n.path_similarity(ts_n)
                    #dist = sense_n.shortest_path_distance(tw_sense_n)
                    #sim = normalization(dist)
                    #print(s_n,ts_n,sim)
                    if not sim:
                        sim=0
                    if sim>max_sim:
                        max_sim=sim

                if title_map[wordslist[i][j]]<max_sim:
                    title_map[wordslist[i][j]] = max_sim

            # if s not in title_map:
            #     title_map[s] = max_sim
    return title_map

#----------------------------视觉信息相关性---------------------

#计算视觉相关性
def visual_sim(wordslist, filter_visual_words):
    visual_map = {}
    for i in range(len(wordslist)):
        words = wordslist[i]
        for w in words:
            if w not in visual_map:
                if w.lower() in filter_visual_words:
                    visual_map[w]=1
                else:
                    visual_map[w]=0

    return visual_map

#----------------------------语义重要性------------------------------------
#输入关键词列表（格式为[[a,b],[c,d],[e]]），根据共现关系，输出一个图
def get_graph(wordslist):
    words = []
    G = nx.Graph()
    for i in wordslist:
        for j in i:
            if j not in words:
                words.append(j)
                G.add_node(j)
    for wl in wordslist:
        for x in range(len(wl)):
            for y in range(x+1,len(wl)):
                G.add_edge(wl[x],wl[y])
    return G

#输入图，计算度中心性，介数中心性，相乘
def semantic_score(G):
    semantic_map = {}
    degree_tuple = G.degree()
    bet_map = nx.betweenness_centrality(G)
    for t in degree_tuple:
        word = t[0]
        degree = t[1]
        bet = bet_map[word]
        score = degree/(len(degree_tuple)-1) * bet
        semantic_map[word]=score
    return semantic_map

#----------------------------计算TF-IDF值------------------------------------
def get_tf(wordslist,txt_path):
    tf_map = {}
    for words in wordslist:
        for word in words:
            tf_map[word]=0
    
    with open(txt_path,'r') as file:
        try:
            data = file.read()
        except:
            with open(txt_path,'r',encoding='gbk') as file:
                data = file.read()

    #分句 ['','','']
    sentences = sent_tokenize(data)
    #分词
    words=[]
    count=0
    for sen in sentences:
        for w in word_tokenize(sen):
            count=count+1
            if w in tf_map:
                tf_map[w]=tf_map[w]+1
    for tf in tf_map:
        tf_map[tf]=tf_map[tf]/count
    return tf_map

def get_idf(wordslist,corpus_txt_path):
    idf_map = {}
    for words in wordslist:
        for word in words:
            idf_map[word]=0
    input_corpus = glob.glob(corpus_txt_path+"/*.txt")
    for cor in input_corpus:
        with open(cor,'r') as file:
            data = file.read()
            for w in idf_map:
                if w in data:
                    idf_map[w]=idf_map[w]+1
    for idf in idf_map:
        idf_map[idf] = math.log( len(input_corpus) / (idf_map[idf]+1) )
    return idf_map
#----------------------------计算总分值，并排序------------------------------------
def compute(title_map,visual_map,semantic_map,tf_map,idf_map):
    total_map = {}
    tfidf_map = {}
    for word in title_map:
        score1 = 3 * title_map[word] + 1 * visual_map[word] + 3 * semantic_map[word] + 2 * tf_map[word]*idf_map[word]
        total_map[word] = score1

        score2 = tf_map[word]*idf_map[word]
        tfidf_map[word] = score2
    return total_map,tfidf_map

######## 以下是RAKE算法代码 ######################################################################################################
###############################################################################################################################
def is_number(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False

def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words

def separate_words(text, min_word_return_size):
    """
    Utility function to return a list of all words that are have a length greater than a specified number of characters.
    @param text The text that must be split in to words.
    @param min_word_return_size The minimum no of characters a word must have to be included.
    """
    splitter = re.compile('[^a-zA-Z0-9_\\+\\-/]')
    words = []
    for single_word in splitter.split(text):
        current_word = single_word.strip().lower()
        #leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
        if len(current_word) > min_word_return_size and current_word != '' and not is_number(current_word):
            words.append(current_word)
    return words

def split_sentences(text):
    """
    Utility function to return a list of sentences.
    @param text The text that must be split in to sentences.
    """
    sentence_delimiters = re.compile(u'[.!?,;:\t\\\\"\\(\\)\\\'\u2019\u2013]|\\s\\-\\s')
    sentences = sentence_delimiters.split(text)
    return sentences

def build_stop_word_regex(stop_word_file_path):
    stop_word_list = load_stop_words(stop_word_file_path)
    stop_word_regex_list = []
    for word in stop_word_list:
        word_regex = r'\b' + word + r'(?![\w-])'  # added look ahead for hyphen
        stop_word_regex_list.append(word_regex)
    stop_word_pattern = re.compile('|'.join(stop_word_regex_list), re.IGNORECASE)
    return stop_word_pattern

def generate_candidate_keywords(sentence_list, stopword_pattern):
    phrase_list = []
    for s in sentence_list:
        tmp = re.sub(stopword_pattern, '|', s.strip())
        phrases = tmp.split("|")
        for phrase in phrases:
            phrase = phrase.strip().lower()
            if phrase != "":
                phrase_list.append(phrase)
    return phrase_list

def calculate_word_scores(phraseList):
    word_frequency = {}
    word_degree = {}
    for phrase in phraseList:
        word_list = separate_words(phrase, 0)
        word_list_length = len(word_list)
        word_list_degree = word_list_length - 1
        #if word_list_degree > 3: word_list_degree = 3 #exp.
        for word in word_list:
            word_frequency.setdefault(word, 0)
            word_frequency[word] += 1
            word_degree.setdefault(word, 0)
            word_degree[word] += word_list_degree  #orig.
            #word_degree[word] += 1/(word_list_length*1.0) #exp.
    for item in word_frequency:
        word_degree[item] = word_degree[item] + word_frequency[item]

    # Calculate Word scores = deg(w)/frew(w)
    word_score = {}
    for item in word_frequency:
        word_score.setdefault(item, 0)
        word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)  #orig.
    #word_score[item] = word_frequency[item]/(word_degree[item] * 1.0) #exp.
    return word_score

def generate_candidate_keyword_scores(phrase_list, word_score):
    keyword_candidates = {}
    for phrase in phrase_list:
        keyword_candidates.setdefault(phrase, 0)
        word_list = separate_words(phrase, 0)
        candidate_score = 0
        for word in word_list:
            candidate_score += word_score[word]
        keyword_candidates[phrase] = candidate_score
    return keyword_candidates

class Rake(object):
    def __init__(self, stop_words_path):
        self.stop_words_path = stop_words_path
        self.__stop_words_pattern = build_stop_word_regex(stop_words_path)

    def run(self, text):
        sentence_list = split_sentences(text)

        phrase_list = generate_candidate_keywords(sentence_list, self.__stop_words_pattern)

        word_scores = calculate_word_scores(phrase_list)

        keyword_candidates = generate_candidate_keyword_scores(phrase_list, word_scores)

        sorted_keywords = sorted(keyword_candidates.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_keywords

def get_key(txt_file,key_file,keylen):
    with open(key_file, "w"):
        pass 

    with open(txt_file,'r') as file:
        try:
            text = file.read()
        except:
            with open(txt_file,'r',encoding='gbk') as file:
                text = file.read()


    stoppath = "SmartStoplist.txt"  #SMART stoplist misses some of the lower-scoring keywords in Figure 1.5, which means that the top 1/3 cuts off one of the 4.0 score words in Table 1.1
    stopwordpattern = build_stop_word_regex(stoppath)

    sentenceList = split_sentences(text)
    # generate candidate keywords
    phraseList = generate_candidate_keywords(sentenceList, stopwordpattern)
    # calculate individual word scores
    wordscores = calculate_word_scores(phraseList)
    # generate candidate keyword scores
    keywordcandidates = generate_candidate_keyword_scores(phraseList, wordscores)
    sortedKeywords = sorted(keywordcandidates.items(), key=operator.itemgetter(1), reverse=True)
    totalKeywords = len(sortedKeywords)
    rake = Rake("SmartStoplist.txt")
    keywords = rake.run(text)
    with open(key_file,'a') as keyfile:
        for k in keywords[:keylen]:
            score = k[1]
            word = k[0]
            keyfile.write(str(word))
            keyfile.write("\n")
######## RAKE算法代码结束 ######################################################################################################
###############################################################################################################################


if __name__ == "__main__":

    input_directory = glob.glob("4education/*")
    for in_dir in input_directory:
        name = in_dir.split('/')[1]
        txt_path = in_dir + "/" + name + "content.txt"
        print(name)

        with open(txt_path,'r') as file:
            try:
                data = file.read()
            except:
                with open(txt_path,'r',encoding='gbk') as file:
                    data = file.read()
        
        #分句 ['','','']
        sentences = sent_tokenize(data)

        #分词和去停等词
        stopwords_path = "english.txt"
        stopwords = stopword_list(stopwords_path)
        wordslist = [remove_stopwords(word_tokenize(sen),stopwords) for sen in sentences]
        print("分句分词完毕")

    #-----------------标题相关性计算---------------------------------------
        title_path = in_dir + "/" + name + "title.txt"
        with open(title_path,'r') as titlefile:
            try:
                title = titlefile.read()
            except:
                with open(title_path,'r',encoding='gbk') as titlefile:
                    title = titlefile.read()

        #标题分词和去停用词
        title_words = word_tokenize(title)
        filter_title_words = remove_stopwords(title_words,stopwords)

        #将词语列表转成语义列表
        senseslist = words_to_senses(wordslist,sentences)

        #将标题词语列表转成语义列表
        title_senses = title_words_to_senses(filter_title_words,title)

        #计算标题相关性得分
        title_map = title_sim(wordslist, senseslist, title_senses)
        print("标题相关性计算完毕")

    #-----------------视觉相关性计算---------------------------------------
        ocr_path = in_dir + "/" + name + "ocr.txt"
        visual_words = []

        with open(ocr_path,'r') as ocrfile:
            try:
                lines = ocrfile.readlines()
            except:
                with open(ocr_path,'r', encoding='gbk') as ocrfile:
                    lines = ocrfile.readlines()
            for line in lines:
                visual_words.append(line.strip())

        filter_visual_words = remove_stopwords(visual_words,stopwords)
        visual_map = visual_sim(wordslist, filter_visual_words)
        print("视觉相关性计算完毕")

    #-----------------语义重要性计算---------------------------------------
        #计算语义重要性得分
        G = get_graph(wordslist)
        semantic_map = semantic_score(G)
        print("语义重要性计算完毕")

    #-----------------TF-IDF值计算---------------------------------------
        tf_map = get_tf(wordslist,txt_path)
        corpus_txt_path = 'newstxt'
        idf_map = get_idf(wordslist,corpus_txt_path)
        print("TF-IDF值计算完毕")

    #-----------------总分值计算---------------------------------------
        total_path = "4education_res/" + name + "my.txt"
        tfidf_path = "4education_res/" + name + "tfidf.txt"
        textrank_path = "4education_res/" + name + "textrank.txt"
        rake_path = "4education_res/" + name + "rake.txt"

        total_map,tfidf_map = compute(title_map,visual_map,semantic_map,tf_map,idf_map)
        total_list = sorted(total_map.items(), key=lambda item:item[1], reverse=True)
        tfidf_list = sorted(tfidf_map.items(), key=lambda item:item[1], reverse=True)

        with open(total_path,'w'),open(tfidf_path,'w'),open(textrank_path,'w'),open(rake_path,'w'):
            pass
        
        with open(total_path,'a') as myfile, open(tfidf_path,'a') as tfidffile:
            wlen=0
            for k in wordslist:
                wlen=wlen+len(k)
            if wlen<5:
                keylen=1
            elif wlen<20:
                keylen=5
            else:
                keylen=10

            for i in range(keylen):
                myw = total_list[i][0]
                tfidfw = tfidf_list[i][0]
                myfile.write(myw)
                myfile.write("\n")

                tfidffile.write(tfidfw)
                tfidffile.write("\n")

        with open(textrank_path,'a') as textrankfile:
            textrankkey = summa.keywords.keywords(data).split('\n')[:keylen]
            for tk in textrankkey:
                textrankfile.write(tk)
                textrankfile.write("\n")


        get_key(txt_path,rake_path,keylen)


