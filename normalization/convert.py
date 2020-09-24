import cv2
import xml.etree.ElementTree as ET
from os import listdir

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

images = listar_imagens("../dataset_yolo/")

# faz um arquivo de texto com os componentes de cada imagem
# e sua posição
for image_name in images:
    image = cv2.imread("../dataset_yolo/"+image_name)
    yolo = open("../dataset_yolo/"+image_name[:-3]+"txt", 'w')

    tree = ET.parse("../dataset_pascal/"+image_name[:-3]+"xml")
    root = tree.getroot()

    for child in root.findall("object"):
        component = child[0].text
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
        yolo.write("0"+" "+str(X)+" "+str(Y)+" "+str(W)+" "+str(H)+"\n")

    yolo.close()