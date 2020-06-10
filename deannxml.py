
import glob
import xml.etree.ElementTree as ET
import os


data_path = 'data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

data_txt_path = 'datatxt'
if not os.path.exists(data_txt_path):
    os.makedirs(data_txt_path)

input_files = glob.glob(data_path+"/*.xml")

# file = 'news/A1E.xml'

#</ce:para></ce:section></ce:sections></body></'article'></xocs:serial-item>
for file in input_files:
    tree = ET.parse(file)
    root = tree.getroot()
    sections = root.find('{http://www.elsevier.com/xml/xocs/dtd}serial-item').find('{http://www.elsevier.com/xml/ja/dtd}article')\
            .find('{http://www.elsevier.com/xml/ja/dtd}body').find('{http://www.elsevier.com/xml/common/dtd}sections')
    for sec in sections.findall('{http://www.elsevier.com/xml/common/dtd}section'):
        for pa in sec.findall('{http://www.elsevier.com/xml/common/dtd}para'):
             print(pa.text)
             for x in pa:
                print(x.text)
                if x.tail:
                    print(x.tail)
        print("\n")


    # namestring = file.split('.')[0]
    # name = namestring.split('/')[-1]
    # txt_file_name = data_txt_path + '/' + name + '.txt'
    # with open(txt_file_name, "w"):
    #         pass

    # for pa in body.iter('ce:para'):
    #     # with open(txt_file_name, "a") as txtfile:
    #     #     txtfile.write(pa.text)
    #     print(pa.text)
