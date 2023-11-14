from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup
from pathlib import Path
import pathlib
import re
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

PATH = "C:\Program Files (x86)\chromedriver.exe"
# driver = webdriver.Chrome(PATH)
service = webdriver.ChromeService(executable_path = PATH)
driver = webdriver.Chrome(service=service)

def scrape_google_images(query, num_images, name):
    # Prepare the query URL
    num_images = num_images - 1
    query = urllib.parse.quote_plus(query)
    url = "https://www.google.com/search?q=" + query + "&source=lnms&tbm=isch"

    # Open the URL in the webdriver
    driver.get(url)

    # Scroll to load more images (optional)
    prev_height = -1
    max_scrolls = 4
    scroll_count = 0

    while scroll_count < max_scrolls:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # give some time for new results to load
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == prev_height:
            break
        prev_height = new_height
        scroll_count += 1

    time.sleep(1.5)  # give some time for new results to load

    # Extract image URLs using BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    img_tags = soup.find_all('img', {'class': 'rg_i'})

    # # Create a directory to save the images
    # os.makedirs("data/{name}", exist_ok=True)

    # Download the images

    count = 0
    for img_tag in img_tags:
        if (img_tag):
            img_url = img_tag['src']
            if img_url:
                count += 1
                image_path = f'data/{name}/{name}{count}.png'
                urllib.request.urlretrieve(img_url, image_path)
                print(f'Downloaded image data/{name}/{name}{count}.png ')


    # Close the webdriver
    driver.quit()

input_dir =  pathlib.Path("data")

folder = input_dir.iterdir()

def cout_data_in_files(files):
    return len(list(files.rglob("*.png*")))
# number_of_data = []
# for files in folder:
#      string = str(files) 
#      number_of_data.append(cout_data_in_files(files))


# print(number_of_data)

mean = 50
print(mean)
for files in folder:
     string = str(files) 
     print(string)
     pattern = r'\\(.*)$'
     name = re.findall(pattern, string)[0]
     if cout_data_in_files(files) < mean:
        scrape_google_images(name, mean, name)
        


