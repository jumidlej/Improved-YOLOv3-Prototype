# Protótipo: Improved YOLOv3 para detecção de componentes eletrônicos em PCBs
Artigo [1]: Application Research of Improved YOLOv3 Algorithm in PCB Electronic Component Detection (2019).

Artigo [2]: YOLOv3: An Incremental Improvement (2018)

## Dataset
Artigo [3]: Data-Efficient Graph Embedding Learning for PCB Component Detection (2018).

* 47 imagens da internet e de câmeras industriais [3]
* Link: https://sites.google.com/view/graph-pcb-detection-wacv19/home

## Label Normalization
Adequação do dataset ao treinamento do algoritmo YOLOv3.

* Normalização dos labels

## Data Normalization
As imagens vão passar por algum tipo de pré-processamento? No modelo do kaggle todas as imagens tem 416x416. No artigo [2] tem 448x448. Ver no [1]. Eu acho que o yolo aceita imagens com qualquer tamanho e faz o resize ele mesmo, mas talvez seja melhor fazer antes.

1. Resize/Rescaling?
2. Deixar só a placa nas imagens? (Há imagens com plano de fundo) Ia ser legal se desse eu acho.

## Data Augmentation
Artigo [4]: The Effectiveness of Data Augmentation in Image Classification using Deep Learning (2017).

Fazer 3 datasets pra ver qual obtém os melhores resultados.

1. Métodos tradiocionais
2. Style transfer
3. Neural Network [4]

Será que fazer um dataset que mescla os métodos obtém bons resultados?

## YOLOv3 no TensorFlow
Kaggle: https://www.kaggle.com/aruchomu/yolo-v3-object-detection-in-tensorflow

GitHub: https://github.com/wizyoung/YOLOv3_TensorFlow