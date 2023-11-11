import pandas as pd 
import requests
import urllib
import urllib.request
import re
from urllib.request import urlopen
from urllib.error import *
import os
def check_url(link):
    try:
            html = urlopen(link)
    except HTTPError as e:
        return False
        
    except URLError as e:
        return False
    else:
        return True

def Enquiry(lis1):
    if len(lis1) == 0:
        return 0
    else:
        return 1

df = pd.read_csv("adidas_nikes_products_snaphost_data.csv")

df1 = df.dropna()
pattern = '(https:\/\/[^~|]*)'

for ind in df1.index:
    url  = re.findall(pattern,df1['images'][ind])
    directory = re.findall("[^]"),df1['name'][ind]
    parent_dir = "data/"
    path = parent_dir + directory
    if (os.path.exists(path) == False):
        os.mkdir(path) 
    print(path) 
    # for index, link in enumerate(url):
    #     if(check_url(link)):
    #         var = requests.get(link)
    #         name = re.findall('([^\/.]*).jpg',link)
    #         if Enquiry(name):
    #             with open('%s/%s.png' % (path,name[0]), 'wb') as f:
    #                 f.write(var.content)
