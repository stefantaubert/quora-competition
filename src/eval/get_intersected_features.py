import json 
import os

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "all.txt")
file_object = open(path, "r")
text = file_object.readlines()

resultingSets = []

for line in text:
    if not line == "\n":
        data = json.loads(line)
        resultingSets.append(set(data))

u = set.intersection(*resultingSets)
u = sorted(u)
# file_object  = open("all.txt", "a")
# for intersectedFeature in u:
#     file_object.write(intersectedFeature + "\n")
# file_object.close()
print(u)