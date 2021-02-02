new_dir = "/home/datasets/"

original = open("train_augmented.txt", "r")
changed = open("train_changed.txt", "w")

for line in original:
    line = line.split()
    line[0] = line[0].split("/")
    line[0] = new_dir+line[0][-1]

    line = " ".join(line)

    changed.write(line+"\n")

original.close()
changed.close()