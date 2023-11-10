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


df = pd.read_csv("adidas_nikes_products_snaphost_data.csv")

df1 = df.dropna()
pattern = '(https:\/\/[^~|]*)'

for image in df1['images']:
    url  = re.findall(pattern,image)
    for index, link in enumerate(url):
        if(check_url(link)):
            var = requests.get(link)
            name = re.findall('([^\/.]*).jpg',link)[0]
            with open('image/%s.png' % (name), 'wb') as f:
                f.write(var.content)
