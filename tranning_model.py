import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np
folder_path = 'data/'

df = pd.read_csv('adidas_nikes_products_snaphost_data.csv')


for index, name in enumerate(df['name']):
    folder_name = folder_path + name
    if (os.path.exists(folder_name)):
        for filename in os.listdir(folder_name):
            file_path = os.path.join(folder_name, filename)
            if os.path.isfile(file_path):
                img = cv2.imread(file_path)