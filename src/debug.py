from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc

def main():

    path2training_data = "../data/training/"
    path2save = "../data/save/"

    filenames_tmp = [f for f in listdir(path2training_data) if isfile(join(path2training_data, f))]
    images = []
    pokemon_names = []
    size = [500,500]
    for filename in filenames_tmp:
    	if filename.split(".")[1] == "png":
            print("here")
            img = Image.open(path2training_data + filename).resize(size)
            img.save(path2save + filename.split(".")[0] + ".png")
            
            img_resized = np.asarray(img).copy()

            for i in range(size[0]):
                for j in range(size[1]):
                    if np.sum(img_resized[i,j,:]) == 0:
                        img_resized[i,j, :] = 255

            img_resized = Image.fromarray(np.uint8(img_resized))

            img_resized = img_resized.convert('RGB')

            img_resized.save(path2save + filename.split(".")[0] + ".jpg")



if __name__ == '__main__':
	main()
