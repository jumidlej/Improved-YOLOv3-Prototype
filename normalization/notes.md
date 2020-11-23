## Label normalization
Adequação do dataset ao treinamento do algoritmo YOLOv3.
Obs: Utilizamos todas em imagens em jpg, mas algumas tem em png também.
* Normalização dos labels
Cada componente possui um nome apenas em todas as images
* Formatação do dataset nos formatos pascal e yolo

## Data Normalization
As imagens vão passar por algum tipo de pré-processamento? No modelo do kaggle todas as imagens tem 416x416. No artigo [4] tem 448x448. Ver no [1]. Eu acho que o yolo aceita imagens com qualquer tamanho e faz o resize ele mesmo, mas talvez seja melhor fazer antes.
* Pré-processamento das imagens?
* Resize/Rescaling?
* Deixar só a placa nas imagens? (Há imagens com plano de fundo) Ia ser legal se desse eu acho.

## Observação
* Nos datasets: ATTIOT_Top.JPG -> ATTIOT_Top.jpg

## Como rodar
* O dataset adquirido possui formatação pascal e converteremos para yolo
* No arquivo convert.py alterar as pastas dos datasets yolo e pascal
* Para visualizar com o labelImg o resultado, os labels e imagens e arquivo classe.txt (formatação yolo) devem estar na mesma pasta