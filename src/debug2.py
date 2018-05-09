from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
import pickle
import cv2

import re

types = ["", 'Psychic', 'Ghost', 'Ground', 'Dragon', 'Rock', 'Grass', 'Normal', 'Bug', 'Fairy', 'Water', 'Fire', 'Fighting', 'Electric', 'Ice', 'Flying', 'Steel', 'Dark', 'Poison']

path2data = "../data/"
with open(path2data + 'name2attributes.pkl', 'rb') as f:
    name2attributes = pickle.load(f)

def main():

    path2data = "../data/"

    fin_name2id = open(path2data + "pokemon.csv")
    lines = fin_name2id.readlines()
    id2attributes = {}
    for i,line in enumerate(lines[1:152]):
        values = line.split(",")
        _id = int(values[0])
        name = values[1]
        # name2id[name] = _id
        attributes = name2attributes[name]
        id2attributes[int(attributes["ID"])] = attributes["Type1"]

    fin_head = open(path2data + "name_head.txt")
    head2type = {}

    fin_tail = open(path2data + "name_tail.txt")
    tail2type = {}

    lines = fin_head.readlines()
    for i, line in enumerate(lines):
        head2type[line[:-1]] = id2attributes[i+1]

    lines = fin_tail.readlines()
    for i, line in enumerate(lines):
        tail2type[line[:-1]] = id2attributes[i+1]



    # path2training_data = "../data/training/image_151/"
    # path2training_data = "./downloaded_img/"
    path2training_data = "../data/mixed_pokemon/"

    filenames = [f for f in listdir(path2training_data) if isfile(join(path2training_data, f))]
    images = []
    pokemon_names = []
    size = [240,240]

    name2attr = {}
    name2type = {}

    for filename in filenames:
    	if filename.split(".")[1] == "jpg":

            pokemon_name = filename.split(".")[0]
            print(pokemon_name)

            for head in head2type.keys():
                if pokemon_name.startswith(head):
                    head_pokemon_type = head2type[head]

            for tail in tail2type.keys():
                if pokemon_name.endswith(tail):
                    tail_pokemon_type = tail2type[tail]

            if head != tail:
                name2type[pokemon_name] = [head_pokemon_type, tail_pokemon_type]
            else:
                name2type[pokemon_name] = head_pokemon_type


    with open(path2data + 'name2type.pkl', 'wb') as f:
        pickle.dump(name2type, f, pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
	main()
