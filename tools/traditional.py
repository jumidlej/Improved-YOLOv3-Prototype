import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return img

def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def rotate(list_ds, data_dir, label_dir, dataset):
    for f in list_ds:
        # pegar a imagem
        image = process_path(f.numpy())
        # imagem alterada esquerda-direita
        flipped = tf.image.rot90(image/255)
        # nome da imagem FORMATO: blahblah/nome.bah
        image_name = str(f.numpy()).split("/")
        image_name = image_name[-1].split(".")[0]
        # salvar na pasta
        tf.keras.preprocessing.image.save_img(
            path=data_dir+'/'+image_name+'-rot.jpg', 
            x=flipped,
            data_format='channels_last',
            file_format='jpeg', 
            scale=True
        )
        # salvar novos parâmetros
        # abrir o arquivo antigo e fazer um novo com o novo nome
        # e com a nova localização dos bounding boxes
        image_txt = open(label_dir+'/'+image_name+'.txt', 'r')
        flipped_txt = open(label_dir+'/'+image_name+'-rot.txt', 'w')
        for line in image_txt:
            line = line.split()
            aux = line[1]
            line[1] = line[2]
            line[2] = str(1-float(aux))
            flipped_txt.write(line[0]+' '+line[1]+' '+line[2]+' '+line[4]+' '+line[3])
            flipped_txt.write("\n")
        image_txt.close()
        flipped_txt.close()

def flip_left_right(list_ds, data_dir, label_dir, dataset):
    for f in list_ds:
        # pegar a imagem
        image = process_path(f.numpy())
        # imagem alterada esquerda-direita
        flipped = tf.image.flip_left_right(image/255)
        # nome da imagem FORMATO: blahblah/nome.bah
        image_name = str(f.numpy()).split("/")
        image_name = image_name[-1].split(".")[0]
        # salvar na pasta
        tf.keras.preprocessing.image.save_img(
            path=data_dir+'/'+image_name+'-h.jpg', 
            x=flipped,
            data_format='channels_last',
            file_format='jpeg', 
            scale=True
        )
        # salvar novos parâmetros
        # abrir o arquivo antigo e fazer um novo com o novo nome
        # e com a nova localização dos bounding boxes
        image_txt = open(label_dir+'/'+image_name+'.txt', 'r')
        flipped_txt = open(label_dir+'/'+image_name+'-h.txt', 'w')
        for line in image_txt:
            line = line.split()
            line[1] = str(1-float(line[1]))
            # line[2] = str(1-float(line[2]))
            i = 1
            for word in line:
                flipped_txt.write(word)
                if i != 5:
                    flipped_txt.write(" ")
                i += 1
            flipped_txt.write("\n")
        image_txt.close()
        flipped_txt.close()

def flip_up_down(list_ds, data_dir, label_dir, dataset):
    for f in list_ds:
        # pegar a imagem
        image = process_path(f.numpy())
        # imagem alterada esquerda-direita
        flipped = tf.image.flip_up_down(image/255)
        # nome da imagem FORMATO: blahblah/nome.bah
        image_name = str(f.numpy()).split("/")
        image_name = image_name[-1].split(".")[0]
        # salvar na pasta
        tf.keras.preprocessing.image.save_img(
            path=data_dir+'/'+image_name+'-v.jpg', 
            x=flipped,
            data_format='channels_last',
            file_format='jpeg', 
            scale=True
        )
        # salvar novos parâmetros
        # abrir o arquivo antigo e fazer um novo com o novo nome
        # e com a nova localização dos bounding boxes
        image_txt = open(label_dir+'/'+image_name+'.txt', 'r')
        flipped_txt = open(label_dir+'/'+image_name+'-v.txt', 'w')
        for line in image_txt:
            line = line.split()
            # line[1] = str(1-float(line[1]))
            line[2] = str(1-float(line[2]))
            i = 1
            for word in line:
                flipped_txt.write(word)
                if i != 5:
                    flipped_txt.write(" ")
                i += 1
            flipped_txt.write("\n")
        image_txt.close()
        flipped_txt.close()

def grayscale(list_ds, data_dir, label_dir, dataset):
    for f in list_ds:
        # pegar a imagem
        image = process_path(f.numpy())
        # imagem alterada esquerda-direita
        flipped = tf.image.rgb_to_grayscale(image/255)
        # nome da imagem FORMATO: blahblah/nome.bah
        image_name = str(f.numpy()).split("/")
        image_name = image_name[-1].split(".")[0]
        # salvar na pasta
        tf.keras.preprocessing.image.save_img(
            path=data_dir+'/'+image_name+'-gray.jpg', 
            x=flipped,
            data_format='channels_last',
            file_format='jpeg', 
            scale=True
        )
        # salvar novos parâmetros
        # abrir o arquivo antigo e fazer um novo com o novo nome
        # e com a nova localização dos bounding boxes
        image_txt = open(label_dir+'/'+image_name+'.txt', 'r')
        flipped_txt = open(label_dir+'/'+image_name+'-gray.txt', 'w')
        for line in image_txt:
            flipped_txt.write(line)
        image_txt.close()
        flipped_txt.close()

def saturation(list_ds, data_dir, label_dir, dataset, factor=3):
    for f in list_ds:
        # pegar a imagem
        image = process_path(f.numpy())
        # imagem alterada esquerda-direita
        flipped = tf.image.adjust_saturation(image/255, factor)
        # nome da imagem FORMATO: blahblah/nome.bah
        image_name = str(f.numpy()).split("/")
        image_name = image_name[-1].split(".")[0]
        # salvar na pasta
        tf.keras.preprocessing.image.save_img(
            path=data_dir+'/'+image_name+'-sat'+str(factor)+'.jpg', 
            x=flipped,
            data_format='channels_last',
            file_format='jpeg', 
            scale=True
        )
        # salvar novos parâmetros
        # abrir o arquivo antigo e fazer um novo com o novo nome
        # e com a nova localização dos bounding boxes
        image_txt = open(label_dir+'/'+image_name+'.txt', 'r')
        flipped_txt = open(label_dir+'/'+image_name+'-sat'+str(factor)+'.txt', 'w')
        for line in image_txt:
            flipped_txt.write(line)
        image_txt.close()
        flipped_txt.close()

def brightness(list_ds, data_dir, label_dir, dataset, factor=0.4):
    for f in list_ds:
        # pegar a imagem
        image = process_path(f.numpy())
        # imagem alterada esquerda-direita
        flipped = tf.image.adjust_saturation(image/255, factor)
        # nome da imagem FORMATO: blahblah/nome.bah
        image_name = str(f.numpy()).split("/")
        image_name = image_name[-1].split(".")[0]
        # salvar na pasta
        tf.keras.preprocessing.image.save_img(
            path=data_dir+'/'+image_name+'-bright'+str(factor)+'.jpg', 
            x=flipped,
            data_format='channels_last',
            file_format='jpeg', 
            scale=True
        )
        # salvar novos parâmetros
        # abrir o arquivo antigo e fazer um novo com o novo nome
        # e com a nova localização dos bounding boxes
        image_txt = open(label_dir+'/'+image_name+'.txt', 'r')
        flipped_txt = open(label_dir+'/'+image_name+'-bright'+str(factor)+'.txt', 'w')
        for line in image_txt:
            flipped_txt.write(line)
        image_txt.close()
        flipped_txt.close()