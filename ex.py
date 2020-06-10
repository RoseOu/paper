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
import base64
from PIL import Image
import networkx as nx


# 科大讯飞语音识别的一些参数设置
STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

#百度文字识别的一些参数设置
BAIDU_API_KEY = "rUzE1WpIkGnuVL6CUKqcBCxr"
BAIDU_SECRET_KEY = "GHItsygUcVKR1tBUxovsMQd6U5GRyr5w"

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

#将视频转换成音频wav
def video_to_audio(path):
    name = path.split(".")[0]
    newpath = name+".wav"
    call(["ffmpeg", "-i", path, newpath])
    return newpath

#输入视频，得到分割的音频
def cut_audio(video,cut_path):
    if not os.path.exists(cut_path):
        os.makedirs(cut_path)
    size = sumTime(video)  #计算视频总长度
    audio = video_to_audio(video)
    flag = 0
    while size - flag*59 > 59:
        flag = flag + 1
        out_name = cut_path+"/"+str(flag)+".wav"
        call(["ffmpeg", "-ss", time_fomat((flag-1)*59), "-i",audio, "-c","copy","-t","59",out_name])

    flag = flag + 1
    out_name = cut_path+"/"+str(flag)+".wav"
    call(["ffmpeg", "-ss", time_fomat((flag-1)*59), "-i",audio, "-c","copy","-t",str(size - (flag-1)*59),out_name])

#将wav_path文件夹里的wav音频全部转换为 16k 16bit 单声道 pcm格式
def wav_to_pcm(wav_path,pcm_path):
    if not os.path.exists(pcm_path):
        os.makedirs(pcm_path)
    input_files = glob.glob(wav_path+"/*.wav")
    length = len(input_files)
    for i in range(length):
        file = wav_path+"/"+str(i+1)+".wav"
        call(["ffmpeg", "-y", "-i", file, "-acodec", "pcm_s16le", "-f", "s16le", "-ac", "1", "-ar", "16000", pcm_path+"/"+str(i+1)+".pcm"])

# —————————— 科大讯飞语音识别生成websoket连接的参数，主要是生成url—————————— #
class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, AudioFile):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.AudioFile = AudioFile

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"domain": "iat", "language":"zh_en","vad_eos":10000} 

    # 生成url
    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        #print("date: ",date)
        #print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        #print('websocket url :', url)
        return url

#将单段语音使用科大讯飞的APi识别为文字
def audio_to_text(audio,index,txt_path):
    wsParam = Ws_Param(APPID='5e5bac9a', APIKey='f943f8b6582fbc4993df77c43f82d7ab',
                       APISecret='145d53e3a1bd16f7d5ae37f26ecedcc3',
                       AudioFile=audio)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    # 收到websocket消息的处理
    def on_message(ws, message):
        try:
            code = json.loads(message)["code"]
            sid = json.loads(message)["sid"]
            if code != 0:
                errMsg = json.loads(message)["message"]
                print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
            else:
                data = json.loads(message)["data"]["result"]["ws"]
                result = ""
            
                with open(txt_path,'a') as file:
                    for i in data:
                        for w in i["cw"]:
                            if w["w"]!="，":
                                result += w["w"]
                    file.write(result)
                    print(result)


        except Exception as e:
            print("receive msg,but parse exception:", e)


    # 收到websocket错误的处理
    def on_error(ws, error):
        print("### error:", error)


    # 收到websocket关闭的处理
    def on_close(ws):
        print("### closed ###")


    # 收到websocket连接建立的处理
    def on_open(ws):
        def run(*args):
            frameSize = 8000  # 每一帧的音频大小
            intervel = 0.04  # 发送音频间隔(单位:s)
            status = STATUS_FIRST_FRAME  # 音频的状态信息，标识音频是第一帧，还是中间帧、最后一帧

            with open(wsParam.AudioFile, "rb") as fp:
                while True:
                    buf = fp.read(frameSize)
                    # 文件结束
                    if not buf:
                        status = STATUS_LAST_FRAME
                    # 第一帧处理
                    # 发送第一帧音频，带business 参数
                    # appid 必须带上，只需第一帧发送
                    if status == STATUS_FIRST_FRAME:
                        d = {"common": wsParam.CommonArgs,
                             "business": wsParam.BusinessArgs,
                             "data": {"status": 0, "format": "audio/L16;rate=16000",
                                      "audio": str(base64.b64encode(buf), 'utf-8'),
                                      "encoding": "raw"}}
                        d = json.dumps(d)
                        ws.send(d)
                        status = STATUS_CONTINUE_FRAME
                    # 中间帧处理
                    elif status == STATUS_CONTINUE_FRAME:
                        d = {"data": {"status": 1, "format": "audio/L16;rate=16000",
                                      "audio": str(base64.b64encode(buf), 'utf-8'),
                                      "encoding": "raw"}}
                        ws.send(json.dumps(d))
                    # 最后一帧处理
                    elif status == STATUS_LAST_FRAME:
                        d = {"data": {"status": 2, "format": "audio/L16;rate=16000",
                                      "audio": str(base64.b64encode(buf), 'utf-8'),
                                      "encoding": "raw"}}
                        ws.send(json.dumps(d))
                        time.sleep(1)
                        break
                    # 模拟音频采样间隔
                    time.sleep(intervel)
            ws.close()
        thread.start_new_thread(run, ())

    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

# 将文件夹内全部语音进行语音识别
def all_audio_to_text(pcm_path,txt_path):
    with open(txt_path, "w"):
        pass 
    input_files = glob.glob(pcm_path+"/*.pcm")
    length = len(input_files)
    for i in range(length):
        pcm = pcm_path+"/"+str(i+1)+".pcm"
        audio_to_text(pcm,i,txt_path)
        with open(txt_path, "a") as file:
            file.write(" ")


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
#讯飞的文字识别，以每秒为单位，提取视频帧的英文文字，并转为小写
def xunfei_pic_to_text(video_path, ocr_path):
    # 印刷文字识别 webapi 接口地址
    URL = "http://webapi.xfyun.cn/v1/service/v1/ocr/general"
    # 应用ID (必须为webapi类型应用，并印刷文字识别服务，参考帖子如何创建一个webapi应用：http://bbs.xfyun.cn/forum.php?mod=viewthread&tid=36481)
    APPID = "5db2564f"
    # 接口密钥(webapi类型应用开通印刷文字识别服务后，控制台--我的应用---印刷文字识别---服务的apikey)
    API_KEY = "d8fc11f2b53faa483dc862dd0a2ae812"
    def getHeader():
    #  当前时间戳
        curTime = str(int(time.time()))
    #  支持语言类型和是否开启位置定位(默认否)
        param = {"language": "cn|en", "location": "false"}
        param = json.dumps(param)
        paramBase64 = base64.b64encode(param.encode('utf-8'))

        m2 = hashlib.md5()
        str1 = API_KEY + curTime + str(paramBase64,'utf-8')
        m2.update(str1.encode('utf-8'))
        checkSum = m2.hexdigest()
    # 组装http请求头
        header = {
            'X-CurTime': curTime,
            'X-Param': paramBase64,
            'X-Appid': APPID,
            'X-CheckSum': checkSum,
            'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8',
        }
        return header

    # 上传文件并进行base64位编码
    visual_words=[]
    video = cv.VideoCapture(video_path)
    fps=int(video.get(5)) #帧速率
    success, frame = video.read()  #VideoCapture得到的图片是RGB空间
    i=1
    while success:
        #if i%fps==0:
        if i%100==0:
            image = cv.imencode('.jpg', frame[..., ::-1])[1].tobytes()   #输入的图片要是一个BGR格式的uint8 ndarray，所以下标要是[..., ::-1]
            img = str(base64.b64encode(image), 'utf-8')
            data = {
                'image': img
                }

            r = requests.post(URL, data=data, headers=getHeader())
            result = str(r.content, 'utf-8')
            js = r.json()
            if js['data'] != '':
                block=js['data']['block']
                for b in block:
                    line = b['line']
                    for l in line:
                        word = l['word']
                        for w in word:
                            content = w['content'].lower()
                            if content not in visual_words:
                                visual_words.append(content)
                                with open(ocr_path,'a') as ocrfile:
                                    ocrfile.write(content)
                                    ocrfile.write("\n")
        success, frame = video.read()    #VideoCapture得到的图片是RGB空间
        i=i+1
    return visual_words

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
    #这里是根据输入关键词词义列表（格式为[[a,b],[c,d],[e]]），阈值alpha
    # for x in range(len(senses)):
    #     if x<len(senses)-1:
    #         x_n = wordnet.synset(senses[x].name())
    #         for y in range(x+1,len(senses)):
    #             y_n = wordnet.synset(senses[y].name())
    #             sim = x_n.path_similarity(y_n)
    #             if not sim:
    #                 sim=0
    #             if sim>=alpha:
    #                 G.add_edge(senses[x],senses[y])
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
        score1 = 0.25 * title_map[word] + 0.25 * visual_map[word] + 0.25 * semantic_map[word] + 0.25 * tf_map[word]*idf_map[word]
        total_map[word] = score1

        score2 = tf_map[word]*idf_map[word]
        tfidf_map[word] = score2
    return total_map,tfidf_map



if __name__ == "__main__":
    #!!!!!!!!!
    title = "The Best Teacher I Never Had"
    video_path = "test6.mp4"

    cut_path = "cut"
    
    cut_audio(video_path,cut_path)
    print("音频切割完毕")
    pcm_path = "pcm"
    
    wav_to_pcm(cut_path,pcm_path)
    print("格式转换完毕")

    txt_path = "news.txt"
    
    #all_audio_to_text(pcm_path,txt_path)
    print("语音识别完毕")

    #分词
    with open(txt_path,'r') as file:
        data = file.read()
    
    #分句 ['','','']
    sentences = sent_tokenize(data)

    #分词和去停等词
    stopwords_path = "english.txt"
    stopwords = stopword_list(stopwords_path)
    wordslist = [remove_stopwords(word_tokenize(sen),stopwords) for sen in sentences]
    print("分句分词完毕")

#-----------------标题相关性计算---------------------------------------
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
    ocr_path = 'ocr.txt'
    with open(ocr_path,'w') as ocrfile:
        pass
    visual_words = xunfei_pic_to_text(video_path,ocr_path)
    print("ocr识别完毕")
    filter_visual_words = remove_stopwords(visual_words,stopwords)
    print(filter_visual_words)
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
    name = video_path.split('.')[0]
    total_path = name + "my.txt"
    tfidf_path = name + "tfidf.txt"
    total_map,tfidf_map = compute(title_map,visual_map,semantic_map,tf_map,idf_map)
    total_list = sorted(total_map.items(), key=lambda item:item[1], reverse=True)
    tfidf_list = sorted(tfidf_map.items(), key=lambda item:item[1], reverse=True)
    
    with open(total_path,'a') as myfile, open(tfidf_path,'a') as tfidffile:
        for i in range(20):
            myw = total_list[i][0]
            tfidfw = tfidf_list[i][0]
            myfile.write(myw)
            myfile.write("\n")

            tfidffile.write(tfidfw)
            tfidffile.write("\n")




    
