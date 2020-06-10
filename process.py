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
import numpy as np
import requests
import base64
from PIL import Image
import shutil  


# 科大讯飞语音识别的一些参数设置
STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


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
    else:
        shutil.rmtree(cut_path)
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
    else:
        shutil.rmtree(pcm_path)
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



if __name__ == "__main__":
    video_path = "speech6.mp4"

    cut_path = "cut"
    
    cut_audio(video_path,cut_path)
    print("音频切割完毕")
    pcm_path = "pcm"
    
    wav_to_pcm(cut_path,pcm_path)
    print("格式转换完毕")

    txt_path = "speech6content.txt"
    
    all_audio_to_text(pcm_path,txt_path)
    print("语音识别完毕")


    ocr_path = 'speech6ocr.txt'
    with open(ocr_path,'w') as ocrfile:
        pass
    visual_words = xunfei_pic_to_text(video_path,ocr_path)
    print("ocr识别完毕")



    
