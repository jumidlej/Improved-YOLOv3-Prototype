# Protótipo: Improved YOLOv3 para detecção de componentes eletrônicos em PCBs
Artigo [1]: Application Research of Improved YOLOv3 Algorithm in PCB Electronic Component Detection (2019).

Artigo [2]: YOLOv3: An Incremental Improvement (2018).

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
* https://medium.com/analytics-vidhya/yolo-v3-introduction-to-object-detection-with-tensorflow-2-ce75749b1c47

## Para treinar um modelo
* Arquivo .txt de rótulos:

resistor

transistor

* Arquivos .txt de treinamento e de teste:

/home/image.jpg xmin,ymin,xmax,ymax,component_number ... xmin,ymin,xmax,ymax,component_number

/home/image.jpg xmin,ymin,xmax,ymax,component_number ... xmin,ymin,xmax,ymax,component_number

## Para treinar a partir do dataset [1] 
### Organização dos arquivos
* Pasta com todos os arquivos xml juntos (labels)
* Pasta com todas as imagens juntas
* Pasta para os arquivos txt que serão gerados (labels)

### Normalizar os labels
* Rodar label_normalization.py

Especificar a pasta com todos os arquivos xml de todas as imagens

* Esse scrip gerará um arquivo chamado classes.txt com todos os componentes encontrados em todas as imagens. 

* Note que existirão alguns que deverão ser excluídos ou estarão repetidos, mas isso será tratado na conversão desses arquivos de rótulos de xml (formatação PASCAL) para txt (formatação YOLO).

### XML to YOLO
* Rodar xml_to_yolo.py

Especificar a pasta de imagens (.jpg)

Especificar a pasta de arquivos xml

Especificar a pasta de arquivos txt que serão gerados

Especificar os componentes excluídos e repetidos a partir do arquivo classes.txt

* Esse script gerará um arquivo txt para cada imagem no formato: nome_da_imagem.txt e dentro desse arquivo cada linha representa um componente nessa imagem no formato: component_number x y width height. Além disso será gerado um arquivo chamado labels.txt com todos os componentes na ordem de seu devido número.

### Data Augmentation in YOLO format (Optional)
* Rodar o jupyter notebook traditional.ipynb

Especificar a pasta com as imagens

Especificar a pasta com os labels (txt)

* Esse script realizará data augmentation e as imagens e labels serão armazenados nas mesmas pastas especificadas.

### YOLO to training format
* Rodar o script yolo_to_training.py

Especificar a pasta com as imagens

Especificar a pasta com os labels (txt)

* Esse script gerará um arquivo chamado train_augmented.txt no seguinte formato:
image.jpg xmin,ymin,xmax,ymax,component_number ... xmin,ymin,xmax,ymax,component_number

## Train
Já possuímos os arquivos necessários para treinar um modelo.

* O arquivo de rótulos: labels.txt
* O arquivo de treinamento: train_augmented.txt
* O arquivo de teste pode ser criado tirando algumas imagens do arquivo de treinamento e colocando em um outro arquivo de teste.

1. git clone https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3.git

2. Dentro do arquivo yolov3/configs.py alterar:
* TRAIN_CLASSES: /home/labels.txt
* TRAIN_ANNOT_PATH: /home/train_augmented.txt
* TEST_ANNOT_PATH: /home/test.txt
* TRAIN_DATA_AUG: False (Como eu fiz data augmentation eu deixei false, mas eu não sei bem o que é isso)

3. Run python3 train.py

Existem várias variáveis importantes para o treinamento que eu ainda não utilizei ou não sei exatamente pra que servem, mas que podem ser alteradas.

## Test
1. Editar (selecionar imagem) e rodar python3 detection_demo.py