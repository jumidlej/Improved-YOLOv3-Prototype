import cv2
import xml.etree.ElementTree as ET
from os import listdir

'''
Argument: Images path (.jpg)
Returns: List of images names
'''
def list_images_names(path=None):
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
Argument: A classes.txt file
Does: A labels.txt file with all components that are not excluded or repeated
Returns: A components dictionary in which components have a respective number
'''
def components_dictionary(classes_file, excluded, repeated):
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

    return components_dict

'''
Arguments: List of images name, dictionary with components and its numbers
Does: A .txt file to every image in yolo format
'''
def pascal_to_yolo_labelImg(yolo_images_path, yolo_labels_path, pascal_labels_path, images_name, components_dict, excluded):
    for name in images_name:
        image = cv2.imread(yolo_images_path+name+".jpg")
        yolo = open(yolo_labels_path+name+".txt", 'w')

        tree = ET.parse(pascal_labels_path+name+".xml")
        root = tree.getroot()

        for child in root.findall("object"):
            component_name = child[0].text
            
            words = component_name.split()
            if len(words)>=2:
                if words[0]+words[1]=='connector"Port':
                    print(component_name)
                    component_name = "connectorPort"
                else:
                    if '"' in words[0]:
                        component_name = words[0].split('"')[1]
                    else:
                        component_name = words[0]
            else:
                component_name = words[0]

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

            if component_name not in excluded:
                yolo.write(str(components_dict[component_name])+" "+str(X)+" "+str(Y)+" "+str(W)+" "+str(H)+"\n")

        yolo.close()
