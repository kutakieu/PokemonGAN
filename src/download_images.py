import urllib.request
import time

path2data = "../data/"
name_head_fin = open(path2data + "name_head.txt")
head_lines = name_head_fin.readlines()
name_tail_fin = open(path2data + "name_tail.txt")
tail_lines = name_tail_fin.readlines()

for i in range(151):
    for j in range(151):

        url = "http://images.alexonsager.net/pokemon/fused/" + str(i+1) + "/" + str(i+1) + "." + str(j+1) + ".png"
        name = head_lines[i][:-1] + tail_lines[j][:-1]
        urllib.request.urlretrieve(url, "./downloaded_img/" + name + ".png")
        time.sleep(0.5)
