from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc

import cv2

def main():

    # path2training_data = "../data/training/image_151/"
    path2training_data = "./downloaded_img/"
    path2save = "../data/mixed_pokemon/"

    filenames_tmp = [f for f in listdir(path2training_data) if isfile(join(path2training_data, f))]
    images = []
    pokemon_names = []
    size = [240,240]
    for filename in filenames_tmp:
    	if filename.split(".")[1] == "png":

            pokemon_name = filename.split(".")[0]
            print(pokemon_name)
            # img = Image.open(path2training_data + filename).resize(size)
            # img = Image.open(path2training_data + filename)
            img_resized = cv2.imread(path2training_data + filename, cv2.IMREAD_UNCHANGED)
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # img.save(path2save + pokemon_name + ".png")
            # print(type(img))

            # img = img.reshape(240,240,4)

            # img_resized = np.asarray(img).copy()
            # print(type(img_resized))
            # print(img_resized[120,120])

            print(img_resized.shape)

            for i in range(img_resized.shape[0]):
                for j in range(img_resized.shape[1]):
                    if np.sum(img_resized[i,j,:]) == 0:
                        img_resized[i,j, :] = 255

            img = Image.fromarray(np.uint8(img_resized))

            img = img.convert('RGB')

            img.save(path2save + pokemon_name + ".jpg")



if __name__ == '__main__':
	main()
