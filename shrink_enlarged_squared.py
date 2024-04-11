#RUN THIS CODE AFTER CHOOSING IMAGE DIMENSIONS AND PATHS BELOW TO GENERATE RESIZED PHOTOS
import cv2
from pathlib import Path
import os

#ADJUST PATHS ACCORDING TO YOUR LOCAL PATHS
directory = "C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/enlarged_squared"
path = 'C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/300x300'


folders_path = []
folders_names = []
x = 1
for fname in os.listdir(directory):

    folder_path = f"{directory}/{fname}"
    folder_path2 = f"{path}/{fname}"
    folders_path.append(folder_path)
    folders_names.append(fname)
    os.makedirs(folder_path2)

i = 1
j = 0
x = 0
for fname in folders_path:
    for img in os.listdir(fname):
        image = cv2.imread(f"{fname}/{img}",0)
        #EDIT IMAGE SIZE HERE
        resized = cv2.resize(image, (300,300))

        cv2.imwrite(os.path.join(f"{path}/{folders_names[j]}" , f"{i}.jpg"), resized)
        i+=1
    j+=1
















