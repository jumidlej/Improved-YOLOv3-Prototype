# Data Augmentation
Fazer 3 datasets pra ver qual obtém os melhores resultados, assim como no artigo [3].
1. Métodos tradiocionais
2. Style transfer
3. Neural Network [3]
Será que fazer um dataset que mescla os métodos obtém bons resultados?

## Etapas
1. Carregar o dataset no TensorFlow
2. Realizar data augmentation e manter os labels

## Traditional Augmentation
Link: https://www.tensorflow.org/tutorials/images/data_augmentation#using_tfimage

1. Rotation (90)
2. Flipping vertically
3. Flipping horizontally
4. Grayscale
5. Saturation
6. Brightness

## Como rodar
* Ter uma pasta com as imagens e uma pasta com os labels na formatação yolo e substituir no tradicional.ipynb
* Para visualizar o resultado com o labelImg, juntar as imagens e labels em uma pasta só

Obs: Rodar apenas uma vez e apenas com as imagens e labels originais 
