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
from PIL import Image
import networkx as nx
import re
import operator
import openpyxl


def read_file(file_path):
    words = []
    with open (file_path,'r') as file:
        lines = file.readlines()
        for line in lines:
            linelist = line.split()
            words.append(linelist)
        return words


def compute(wordslist, gt):
    inx = []
    flag = 0
    front = []
    for i in range(len(wordslist)):
        l = [k for k in wordslist[i]]
        for word in l:
            for j in range(i):
                if word in wordslist[j]:
                    wordslist[i].remove(word)


    for ph in wordslist:
        flag = 0
        for w in ph:
            if flag:
                break
            else:
                for gtph in gt:
                    if flag:
                        break
                    else:
                        for wgt in gtph:
                            if flag:
                                break
                            if not flag:
                                if w.lower() == wgt.lower():

                                    inx.append(gt.index(gtph))
                                    flag = 1
    p = len(inx)
    r = len(set(inx))
    klen = len(wordslist)
    return p,r,klen


if __name__ == "__main__": 
    excel_path = "/Users/rose/Desktop/test.xlsx"
    workbook = openpyxl.load_workbook(excel_path)
    res = workbook.worksheets[0]
    nrows = res.rows
    ncols = res.columns
    keys = []

    input_directory = glob.glob("4education/*")
    for in_dir in input_directory:
        name = in_dir.split('/')[1]
        gt_path = in_dir + "/" + name + "gt.txt"
        gt = read_file(gt_path)
        my_path = "4education_res/" + name + "my.txt"
        my = read_file(my_path)
        rake_path = "4education_res/" + name + "rake.txt"
        rake = read_file(rake_path)
        textrank_path = "4education_res/" + name + "textrank.txt"
        textrank = read_file(textrank_path)
        tfidf_path = "4education_res/" + name + "tfidf.txt"
        tfidf = read_file(tfidf_path)

        gtlen = len(gt)
        myp, myr, mylen = compute(my, gt)
        rakep, raker, rakelen = compute(rake, gt)
        textrankp, textrankr, textranklen = compute(textrank, gt)
        tfidfp, tfidfr, tfidflen = compute(tfidf, gt)
        keylist = [name, gtlen, myp,myr,mylen,rakep, raker,rakelen,textrankp,textrankr,textranklen,tfidfp, tfidfr,tfidflen]
        keys.append(keylist)
    


    for i in range(1,15):
        for j in range(2,2+len(keys)):
            res.cell(j, i).value = keys[j-2][i-1]
    
    workbook.save(excel_path)
    




