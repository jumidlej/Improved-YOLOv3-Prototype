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

zener

* Arquivos .txt de treinamento e de teste:

/home/image.jpg xmin,ymin,xmax,ymax,component_number ... xmin,ymin,xmax,ymax,component_number

/home/image.jpg xmin,ymin,xmax,ymax,component_number ... xmin,ymin,xmax,ymax,component_number

* Caso haja algum problema com exaustão de memória da placa de vídeo, recomendo tentar sem utilizar a placa.

## Para treinar a partir do dataset [3] 
É preciso ter as imagens originais e os labels (arquivos xml) em pastas separadas e especificar uma pasta vazia para criação dos labels (arquivos txt).
Link para Google Drive com o dataset e as pastas já criadas https://drive.google.com/drive/folders/1Q8m0afWvAhBoQz5nKXHkiOnsoBIg0jYi?usp=sharing.
Pra ficar mais organizado eu recomendo copiar as imagens da pasta pascal/images para pasta yolo/images também.

### Organização dos arquivos
* Pasta com todos os arquivos xml (labels)
* Pasta com todas as imagens
* Pasta para os arquivos txt que serão gerados (labels)

### Criação dos arquivos necessários para treinamento
* Rodar tools/pcb2019_to_train.py

Especificar a pasta de imagens (.jpg) (yolo/images)

Especificar a pasta de arquivos xml existentes (pascal/labels)

Especificar a pasta de arquivos txt que serão gerados (yolo/labels)

Verificar se quer realizar data augmentation ou não

* Pode levar alguns minutos, para verificar se está funcionando é possível verificar se o número de imagens e labels está aumentando nas respectivas pastas.

* Note que data augmentation só deve ser realizado uma vez. Caso ocorra algum erro é necessário deixar na pasta de imagens somente as 47 imagens originais e rodar o código novamente. Ao final do processo devem haver 1504 imagens.

### Explicação das etapas
1. Normalizar os labels
* Arquivo tools/label_normalization.py

Especificar a pasta com todos os arquivos xml de todas as imagens

* Esse scrip gerará um arquivo chamado classes.txt com todos os componentes encontrados em todas as imagens. 

* Note que existirão alguns que deverão ser excluídos ou estarão repetidos, mas isso será tratado na conversão desses arquivos de rótulos de xml (formatação PASCAL) para txt (formatação YOLO).

2. XML to YOLO
* Arquivo tools/xml_to_yolo.py

Especificar a pasta de imagens (.jpg)

Especificar a pasta de arquivos xml

Especificar a pasta de arquivos txt que serão gerados

* Esse script gerará um arquivo txt para cada imagem (nome_da_imagem.txt) e dentro desse arquivo cada linha representará um componente dessa imagem no formato: component_number x y width height. Além disso será gerado um arquivo chamado labels.txt com todos os componentes de todas as imagens.

3. Data Augmentation in YOLO format (Optional)
* Arquivo augmentation/traditional.ipynb

Especificar a pasta com as imagens

Especificar a pasta com os labels (txt)

* Esse script realizará data augmentation e as imagens e labels serão armazenados nas mesmas pastas especificadas.

4. YOLO to training format
* Arquivo tools/script yolo_to_training.py

Especificar a pasta com as imagens

Especificar a pasta com os labels (txt)

* Esse script gerará um arquivo chamado train_augmented.txt no seguinte formato:
image.jpg xmin,ymin,xmax,ymax,component_number ... xmin,ymin,xmax,ymax,component_number

## Treinamento
Já possuímos os arquivos necessários para treinar um modelo.

* O arquivo de rótulos: labels.txt
* O arquivo de treinamento: train_augmented.txt
* O arquivo de teste pode ser criado copiando algumas linhas do arquivo de treinamento e colando em um outro arquivo txt de teste.

1. Dentro do arquivo yolov3/configs.py alterar:
* TRAIN_CLASSES: dir/labels.txt
* TRAIN_ANNOT_PATH: dir/train_augmented.txt
* TEST_ANNOT_PATH: dir/test.txt
* TRAIN_DATA_AUG: False (Caso já tenha realizado data augmentation pode colocar False, mas eu não sei bem o que é isso pra falar a verdade)

2. Run python3 train.py

Existem várias variáveis importantes para o treinamento que eu ainda não utilizei ou não sei exatamente pra que servem, mas que podem ser alteradas.

## Teste
1. Editar (selecionar imagem) e rodar python3 detection_demo.py
