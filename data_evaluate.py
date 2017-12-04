#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:01:57 2017

@author: qifeng
"""
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import re
import csv
import numpy as np
import pandas as pd
#url_sample = 'https://www.google.com/search?q=bitcoin&biw=1200&bih=723&source=lnt&tbs=cdr%3A1%2Ccd_min%3A11%2F6%2F2017%2Ccd_max%3A11%2F9%2F2017&tbm=nws'
'''this is for url string link'''
url_1 = 'https://www.google.com/search?q=' #append keyword
url_2 = '&biw=1200&bih=723&source=lnt&tbs=cdr%3A1%2Ccd_min%3A' #append start month
url_3 = '%2F' #append day and year
url_4 = '%2Ccd_max%3A'#append end month
url_5 = '&tbm=nws'

df2 = pd.read_csv('date_try.csv')
df2 = np.array(df2)
#print(df2[:,0])
row1 = df2[:,0] #get a start date string list
row2 = df2[:,1] #get end date string list
label = []
#result = []
#print(date1[1])
'''get year month day'''
for i in range(3):
    date1 = row1[i].split('/')
    year1 = date1[2].split(' ')
    start_y = '20' + year1[0]
    start_m = date1[0]
    start_d = date1[1]
    date2 = row2[i].split('/')
    year2 = date2[2].split(' ')
    end_y = '20' + year2[0]
    end_m = date2[0]
    end_d = date2[1]
    '''calcualte the duration of target period'''
    if int(start_m) == int(end_m): 
        dura = int(end_d) - int(start_d) + 1
    else:
        dura = 30 - int(start_d) + int(end_d) + 1
    #print(dura)
    url = url_1 + 'bitcoin' + url_2 + start_m + url_3 + start_d + url_3 + start_y + url_4 + end_m + url_3 + end_d + url_3 + end_y + url_5
    url_whole = url_1 + 'bitcoin' + url_2 + '01' + url_3 + '01' + url_3 + start_y + url_4 + '12' + url_3 + '31' + url_3 + end_y + url_5
    req_1 = Request(url, headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'})
    req_2 = Request(url_whole, headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'})
    html_1 = urlopen(req_1)
    html_2 = urlopen(req_2)
    soup_1 = BeautifulSoup(html_1.read())
    soup_2 = BeautifulSoup(html_2.read())
    pid_1 = soup_1.find('div',id = 'resultStats')
    pid_2 = soup_2.find('div',id = 'resultStats')
    result_1 = pid_1.text.split()[1]    #get result number string
    result_2 = pid_2.text.split()[1]
    result_1 = int(re.sub("\D", "", pid_1.text.split()[1])) #transfer number string to int (delete comma)
    result_2 = int(re.sub("\D", "", pid_2.text.split()[1]))
    #result.append(result_1)
    #print(result_2 * dura/365)
    if(result_1 > result_2 * dura /365):
        #print(1)
        label.append(1)
    else:
        #print(0)
        label.append(0)
label = np.array(label)
#result = np.array(result)
with open('labels_2.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Y_label"])
    writer.writerows(map(lambda x: [x],label))
