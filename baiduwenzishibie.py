#获取百度的access_token，有效期一个月
def get_access_token():
    # headers = {
    #     'Content-Type': 'application/json;charset=UTF-8'
    # }
    # res = requests.get(url=host, headers=headers).json()
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+BAIDU_API_KEY+'&client_secret='+BAIDU_SECRET_KEY
    response = requests.get(host)
    if response:
        res=response.json()
        access_token=res['access_token']
        print(access_token)
    else:
        print("Fail to get access token")
    return access_token

def pic_to_text(access_token):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"

    # 读帧
    # success, frame = video.read()    #VideoCapture得到的图片是RGB空间
    # image = cv.imencode('.jpg', frame[..., ::-1])[1].tobytes()   #输入的图片要是一个BGR格式的uint8 ndarray，所以下标要是[..., ::-1]
    # img = base64.b64encode(image)
    # image=open('47.jpg','rb').read()
    # img = base64.b64encode(image)

    # params = {"image":img}
    # request_url = request_url + "?access_token=" + access_token
    # headers = {'content-type': 'application/x-www-form-urlencoded'}
    # response = requests.post(request_url, data=params, headers=headers)
    # if response:
    #     print (response.json())
    video = cv.VideoCapture(video_path)
    success, frame = video.read()  #VideoCapture得到的图片是RGB空间
    i=1
    while success:
        #address = str(i)+ '.jpg'
        #cv.imwrite(address,frame)
        image = cv.imencode('.jpg', frame[..., ::-1])[1].tobytes()   #输入的图片要是一个BGR格式的uint8 ndarray，所以下标要是[..., ::-1]
        img = base64.b64encode(image)
        params = {"image":img,"language_type":"ENG"}
        request_url = request_url + "?access_token=" + access_token
        headers = {'content-type': 'application/x-www-form-urlencoded'}
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            print(i)
            print(response.text)
            #print(response.json())
        success, frame = video.read()    #VideoCapture得到的图片是RGB空间
        i=i+1
        if i==40:
            time.sleep(15)



# #计算某一词语的标题相关性（先将该词语变成词义，然后计算该词义与每一标题词的词义的相似度，最后相加）
# def word_title_sim(word, sentence, title_words, title):
#     #all_sense = wordnet.synsets(word)
#     sense = simple_lesk(sentence, word) #可指定pos='n'
#     if not sense:
#         #return float("inf")
#         return 0

#     sum_sim = 0
#     for tw in title_words:
#         tw_sense = simple_lesk(title, tw) #可指定pos='n'
#         if not tw_sense:
#             continue
#         else:
#             sense_n = wordnet.synset(sense.name())
#             tw_sense_n = wordnet.synset(tw_sense.name())

#             #dist = sense_n.shortest_path_distance(tw_sense_n)
#             #sim = normalization(dist)
#             sim = sense_n.path_similarity(tw_sense_n)
#             sum_sim = sum_sim+sim
#     return sum_sim






# file = input_files[0]
file = 'A1E.xml'

tree = ET.parse(file)
root = tree.getroot()

wtext = root.find('wtext')
for child in wtext:
    print(child.tag, child.attrib)