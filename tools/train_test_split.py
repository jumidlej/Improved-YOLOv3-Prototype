import random

test = 0.2

ds = open("train_changed.txt", "r")
train_file = open("train_split.txt", "w")
test_file = open("test_split.txt", "w")

lines = ds.readlines()

random.shuffle(lines)

for i in range(len(lines)):
    if i < len(lines)*0.2:
        test_file.write(lines[i])
    else:
        train_file.write(lines[i])

ds.close()
train_file.close()
test_file.close()

