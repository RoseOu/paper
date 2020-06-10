
import glob
import xml.etree.ElementTree as ET
import os


news_path = 'news'
if not os.path.exists(news_path):
    os.makedirs(news_path)

news_txt_path = 'newstxt'
if not os.path.exists(news_txt_path):
    os.makedirs(news_txt_path)

input_files = glob.glob(news_path+"/*.xml")

# file = 'news/A1E.xml'
for file in input_files:
    tree = ET.parse(file)
    root = tree.getroot()
    wtext = root.find('wtext')

    namestring = file.split('.')[0]
    name = namestring.split('/')[-1]
    txt_file_name = news_txt_path + '/' + name + '.txt'
    with open(txt_file_name, "w"):
            pass

    for w in wtext.iter('w'):
        with open(txt_file_name, "a") as txtfile:
            txtfile.write(w.text)
        print(w.text)
