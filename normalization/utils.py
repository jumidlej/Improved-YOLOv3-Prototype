import cv2
import xml.etree.ElementTree as ET
from os import listdir

yolo_labels_path = "/home/jacq/Documentos/Datasets/pcb_yolo/"
pascal_labels_path = "/home/jacq/Documentos/Datasets/pcb_pascal/labels/"
train_labels_path = "/home/jacq/Documentos/Datasets/pcb_train/labels/"
augmented_labels_path = "/home/jacq/Documentos/Datasets/pcb_augmented/labels/"

yolo_images_path = "/home/jacq/Documentos/Datasets/pcb_yolo/"
pascal_images_path = "/home/jacq/Documentos/Datasets/pcb_pascal/images/"
train_images_path = "/home/jacq/Documentos/Datasets/pcb_train/images/"
augmented_images_path = "/home/jacq/Documentos/Datasets/pcb_augmented/images/"

'''
Argument: Images path (.jpg)
Returns: List of images name
'''
def list_images_name(path=None):
    if path == None:
        print("Nenhuma pasta foi especificada.")
        return 0

    images = []
    files = [f for f in listdir(path)]
    for f in files:
        if f[len(f)-4:] == ".jpg":
            images.append(f[:-4])

    return images

'''
Argument: List of images name, dictionary with components and its numbers, 
    list of excluded components
Does: A .txt file to every image in yolo format
'''
def pascal_to_yolo_labelImg(images_name, components_dict, excluded_components):
    for name in images_name:
        image = cv2.imread(yolo_images_path+name+".jpg")
        yolo = open(yolo_labels_path+name+".txt", 'w')

        tree = ET.parse(pascal_labels_path+name+".xml")
        root = tree.getroot()

        for child in root.findall("object"):
            component_name = child[0].text

            quotes = component_name.split('"')
            if len(quotes)>=3 and quotes[0]=="connector " and quotes[1].split()[0]=="Port":
                component_name = "connector Port"
                # print(component_name)
            elif len(quotes)>=3 and component_name[0]=='"':
                component_name = quotes[1]
            else:
                component_name = component_name.split()[0]

            xmin = int(child[4][0].text)
            ymin = int(child[4][1].text)
            xmax = int(child[4][2].text)
            ymax = int(child[4][3].text)

            #print(image.shape)
            px = image.shape[1]
            py = image.shape[0]

            # cálculo pro quadradinho ser igual o do yolo
            # ponto no centro, altura e largura, dividos pelo tamanho da imagem
            X = ((xmax+xmin)/2)/px
            Y = ((ymax+ymin)/2)/py
            W = (xmax-xmin)/px
            H = (ymax-ymin)/py

            if component_name not in excluded_components:
                yolo.write(str(components_dict[component_name])+" "+str(X)+" "+str(Y)+" "+str(W)+" "+str(H)+"\n")

        yolo.close()

'''
Argument: List of images name, dictionary with components and its numbers, 
    list of excluded components
Does: A .txt file with every image and its bounding boxes in this format:
    image xmin,ymin,xmax,ymax,class_number xmin,ymin,xmax,ymax,class_number ...
'''
def pascal_to_training(images_name, components_dict, excluded_components):
    train_file = open("train.txt", 'w')
    for name in images_name:
        train_file.write(train_images_path+name+".jpg")

        tree = ET.parse(pascal_labels_path+name+".xml")
        root = tree.getroot()

        for child in root.findall("object"):
            component_name = child[0].text

            quotes = component_name.split('"')
            if len(quotes)>=3 and quotes[0]=="connector " and quotes[1].split()[0]=="Port":
                component_name = "connector Port"
                # print(component_name)
            elif len(quotes)>=3 and component_name[0]=='"':
                component_name = quotes[1]
            else:
                component_name = component_name.split()[0]

            xmin = int(child[4][0].text)
            ymin = int(child[4][1].text)
            xmax = int(child[4][2].text)
            ymax = int(child[4][3].text)

            if component_name not in excluded_components:
                train_file.write(" "+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","+str(components_dict[component_name]))
        train_file.write("\n")
    train_file.close()

'''
Argument: List of images name, dictionary with components and its numbers, 
    list of excluded components
Does: A .txt file to every image with this format:
    xmin, ymin, xmax, ymax, class_number
    in the folder "../augmentation/labels"
'''
def pascal_to_txt(images_name, components_dict, excluded_components):
    for name in images_name:
        train_file = open("../augmentation/labels/"+name+".txt", 'w')

        tree = ET.parse(pascal_labels_path+name+".xml")
        root = tree.getroot()

        for child in root.findall("object"):
            component_name = child[0].text

            quotes = component_name.split('"')
            if len(quotes)>=3 and quotes[0]=="connector " and quotes[1].split()[0]=="Port":
                component_name = "connector Port"
            elif len(quotes)>=3 and component_name[0]=='"':
                component_name = quotes[1]
            else:
                component_name = component_name.split()[0]

            xmin = int(child[4][0].text)
            ymin = int(child[4][1].text)
            xmax = int(child[4][2].text)
            ymax = int(child[4][3].text)

            if component_name not in excluded_components:
                train_file.write(str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+str(components_dict[component_name]))
                train_file.write("\n")
        train_file.close()

'''
Does: A .txt file with every image in the augmented_images_path in this format:
    image xmin,ymin,xmax,ymax,class_number xmin,ymin,xmax,ymax,class_number ...
'''
def yolo_augmented_to_training():
    images_name = []
    files = [f for f in listdir(augmented_labels_path)]
    for f in files:
        if f[len(f)-4:] == ".txt":
            images_name.append(f[:-4])

    train_file = open("train_augmented.txt", 'w')
    for name in images_name:
        train_file.write(augmented_images_path+name+".jpg")
        image = cv2.imread(augmented_images_path+name+".jpg")

        labels_file = open(augmented_labels_path+name+".txt", "r")

        for line in labels_file:
            line = line.split()

            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])

            px = image.shape[1]
            py = image.shape[0]

            xmin = int((x-w/2)*px)
            ymin = int((y-h/2)*py)
            xmax = int((x+w/2)*px)
            ymax = int((y+h/2)*py)

            train_file.write(" "+str(xmin)+","+str(ymin)+","+str(xmax)+","+str(ymax)+","+line[0])
        train_file.write("\n")
    train_file.close()

'''
Does: Test if the bounding boxes are correct in the training augmented file
Problem: Basicly never stops so you have to kill it
'''
def test_yolo_augmented_to_training():
    train = open("train_augmented.txt", "r")
    for line in train:
        line = line.split()
        image = cv2.imread(line[0])
        for i in range(1, len(line)):
            line[i]=line[i].split(",")
            xmin = int(line[i][0])
            ymin = int(line[i][1])
            xmax = int(line[i][2])
            ymax = int(line[i][3])
            image = cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
        
        scale_percent = 60 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Imagem", resized)
        cv2.waitKey(0)

'''
Argument: Name of xml file
Returns: Set with all the components of this file
'''
def naive_normalization(name):
    tree = ET.parse(pascal_labels_path+name+".xml")
    root = tree.getroot()

    components = []

    for child in root.findall("object"):
        components.append(child[0].text)

    # verifica: se tem aspas -> pega o que tem dentro
    # se não for, verifica se é connector Port
    # se não for, pega a primeira palavra
    for i in range(len(components)):
        quotes = components[i].split('"')
        if len(quotes)>=3 and quotes[0]=="connector " and quotes[1].split()[0]=="Port":
            components[i] = "connector Port"
            # print(components[i])
        elif len(quotes)>=3 and components[i][0]=='"':
            components[i] = quotes[1]
        else:
            components[i] = components[i].split()[0]

    components = set(components)

    # print(components)
    
    return components

'''
Argument: List of images name
Does: A classes.txt file with all components of all the images
'''
def normalization(images_name):
    # create a empty set
    components = set()
    for name in images_name:
        components = components.union(naive_normalization(name))
    
    # print(components)
    components = list(components)
    components.sort()

    classes = open("classes.txt", "w")
    for component in components:
        classes.write(component+"\n")

    classes.close()

'''
Argument: A classes.txt file
Does: A labels.txt file with all components that are not excluded or repeated
Returns: A components dictionary in which components have a respective number, 
    a list of excluded components
'''
def classes(classes_file):
    # componentes excluidos
    excluded = ["text", "component text"]
    repeated = ["capacitor jumper", "resistor jumper", "resistor network", "diode zener array"]

    components_dict = {}

    classes = open(classes_file, "r")
    labels = open("labels.txt", "w")
    component_number = 0
    for component in classes:
        if component[:-1] not in excluded:
            if component[:-1] in repeated:
                component_number -= 1
                components_dict[component[:-1]] = component_number
            else:
                components_dict[component[:-1]] = component_number
                labels.write(component)
            component_number += 1

    labels.close()
    classes.close()

    return components_dict, excluded

# images_name = list_images_name(pascal_images_path)
# print(images_name)
# print(len(images_name))

# normalization(images_name)
# components_dict, excluded = classes("classes.txt")
# print(components_dict)

# pascal_to_yolo_labelImg(images, components_dict, excluded)
# pascal_to_yolo_training(images_name, components_dict, excluded)
# pascal_to_txt(images_name, components_dict, excluded)
# yolo_augmented_to_training()
test_yolo_augmented_to_training()