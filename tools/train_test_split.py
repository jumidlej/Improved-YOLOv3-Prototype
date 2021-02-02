import random
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-sp", "--split", type=float,
	help="Test size  0.0-1.0",  required=True)
ap.add_argument("-sz", "--size", type=int,
	help="Data size",  required=True)
args = vars(ap.parse_args())

txTest = args["split"]
szTest = args["size"]

ds = open("train_augmented.txt", "r")
train_file = open("train_split.txt", "w")
test_file = open("test_split.txt", "w")

lines = ds.readlines()

if(len(lines) < szTest):
  szTest = len(lines)

random.shuffle(lines)

for i in range(szTest):
    if i < szTest*0.4:
        test_file.write(lines[i])
    else:
        train_file.write(lines[i])

ds.close()
train_file.close()
test_file.close()