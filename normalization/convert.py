import cv2
import xml.etree.ElementTree as ET
from os import listdir

yolo_labels_path = "/home/jacq/Documentos/Datasets/pcb_yolo/"
pascal_labels_path = "/home/jacq/Documentos/Datasets/pcb_pascal/labels/"
train_labels_path = "/home/jacq/Documentos/Datasets/pcb_train/labels/"

yolo_images_path = "/home/jacq/Documentos/Datasets/pcb_yolo/"
pascal_images_path = "/home/jacq/Documentos/Datasets/pcb_pascal/images/"
train_images_path = "/home/jacq/Documentos/Datasets/pcb_train/images/"

'''
Argument: Images path (.jpg)
Returns: List of images name
'''
def list_image_names(path=None):
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
def pascal_to_yolo_labelImg(image_names, components_dict, excluded_components):
    for name in image_names:
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
def pascal_to_yolo_training(image_names, components_dict, excluded_components):
    train_file = open("train.txt", 'w')
    for name in image_names:
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
def pascal_to_txt(image_names, components_dict, excluded_components):
    for name in image_names:
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
def normalization(image_names):
    # create a empty set
    components = set()
    for name in image_names:
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

image_names = list_image_names(pascal_images_path)
print(image_names)
print(len(image_names))

normalization(image_names)
components_dict, excluded = classes("classes.txt")
print(components_dict)

# pascal_to_yolo_labelImg(images, components_dict, excluded)
# pascal_to_yolo_training(image_names, components_dict, excluded)
pascal_to_txt(image_names, components_dict, excluded)