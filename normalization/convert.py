import cv2
import xml.etree.ElementTree as ET
from os import listdir

yolo_path = "/home/ju/Documentos/projeto/Improved-YOLOv3-Prototype/dataset_yolo/"
pascal_path = "/home/ju/Documentos/projeto/Improved-YOLOv3-Prototype/dataset_pascal/"

# lista todas em imagens com extensão .jpg em uma pasta
def listar_imagens(path=None):
    if path == None:
        return 0

    images = []
    files = [f for f in listdir(path)]
    for f in files:
        #print(f[len(f)-3:])
        if f[len(f)-3:] == "jpg":
            images.append(f)

    return images

# cria um arquivo .txt para cada imagem/.yml com a posição dos componentes
def pascal_to_yolo(yolo_path, pascal_path, images_name, components_dict, excluded_components):
    for name in images_name:
        image = cv2.imread(yolo_path+name)
        yolo = open(yolo_path+name[:-3]+"txt", 'w')

        tree = ET.parse(pascal_path+name[:-3]+"xml")
        root = tree.getroot()

        for child in root.findall("object"):
            component_name = child[0].text

            quotes = component_name.split('"')
            if (len(quotes)>=3) and component_name[0]=='"':
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

            # falta saber identificar o componente pelo número
            # yolo.write(component+" "+str(X)+" "+str(Y)+" "+str(W)+" "+str(H)+"\n")
            if component_name not in excluded_components:
                yolo.write(str(components_dict[component_name])+" "+str(X)+" "+str(Y)+" "+str(W)+" "+str(H)+"\n")

        yolo.close()

# aqui a gente devia arrumar os componentes
# pegar todos os nomes de componentes de cada imagem
# colocar em uma lista
# ordenar em ordem alfabetica
# veriricar se existem componentes com duas palavras iguais
# verificar se existem componentes com uma palavra igual
# retorna um conjunto com os nomes de componentes
# PROBLEMAS:
# E se forem três palavras iguais ou mais
# E se existir esse caso:
# component text C3
# component text C4
# component A
def image_normalization(image_file):
    tree = ET.parse(image_file[:-3]+"xml")
    root = tree.getroot()

    components = []

    # nome de todos os componentes do arquivo xml
    for child in root.findall("object"):
        components.append(child[0].text)

    # ordenação
    components.sort()

    # faz dois a dois em toda lista de componentes:
    # verifica se as duas primeiras palavras são iguais, se não
    # verifica se as primeiras palavras são iguais
    for i in range(len(components)-1):
        # testar se as duas primeiras palavras são iguais se existir mais de uma palavra
        # exceções: unknown
        # print("componente "+components[i])
        # print("componente "+components[i+1])
        d1 = components[i].split()[0]
        d2 = components[i+1].split()[0]
        if (len(components[i].split())>1):
            c1 = components[i].split()[0]+" "+components[i].split()[1]
            c2 = components[i+1].split()[0]+" "+components[i+1].split()[1]
            # print("c1"+c1)
            # print("c2"+c2)
            # print("d1"+d1)
            # print("d2"+d2)
            if (c1 == c2):
                components[i] = c1
                components[i+1] = c1
            elif (d1 == d2):
                components[i] = d1
                components[i+1] = d1
        # se não testar se a primeira palavra é igual
        elif (d1 == d2):
            components[i] = d1
            components[i+1] = d1

    components = set(components)

    # for component in components:
    #    print(component)

    return components

# Se tem aspas são duas palavras
# Se não, pega apenas a primeira
# retorna um conjunto com os nomes de componentes
def naive_normalization(image_file):
    tree = ET.parse(image_file[:-3]+"xml")
    root = tree.getroot()

    components = []

    for child in root.findall("object"):
        components.append(child[0].text)

    for i in range(len(components)):
        quotes = components[i].split('"')
        if (len(quotes)>=3) and components[i][0]=='"':
            components[i] = quotes[1]
        else:
            components[i] = components[i].split()[0]
    
    components = set(components)

    # for component in components:
    #    print(component)
    
    return components

# unir todos os conjuntos de componentes de todas as imagens
# colocar cada componente e seu número em um arquivo
# criar um dicionario com o componente e seu numero
def normalization(images_name):
    # componentes excluidos
    excluded = ["text", "component text", "unknown"]
    # create a empty set
    components = set()
    for image in images_name:
        #image_normalization(pascal_path+image)
        components = components.union(naive_normalization(pascal_path+image))
        # print(naive_normalization(pascal_path+image))
    
    # print(components)
    components = list(components)
    components.sort()

    components_dict = {}

    classes = open("classes.txt", "w")
    component_number = 0
    for component in components:
        if component not in excluded:
            classes.write(component+"\n")
            components_dict[component] = component_number
            component_number += 1

    classes.close()

    return components_dict, excluded

images = listar_imagens(pascal_path)
# a = image_normalization(pascal_path+images[0])
# b = naive_normalization(pascal_path+images[0])
# print("A - B\n"+str(a.difference(b)))
# print("B - A\n"+str(b.difference(a)))
components_dict, excluded = normalization(images)
# print(components_dict)
# print(excluded)
pascal_to_yolo(yolo_path, pascal_path, images, components_dict, excluded)