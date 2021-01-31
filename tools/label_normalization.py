import xml.etree.ElementTree as ET
from os import listdir

'''
Argument: Labels path (.xml)
Returns: List of xml files
'''
def list_xml_files(path=None):
    if path == None:
        print("Nenhuma pasta foi especificada.")
        return 0

    xml_files = []
    files = [f for f in listdir(path)]
    for f in files:
        if f[len(f)-4:] == ".xml":
            xml_files.append(f)

    return xml_files

'''
Argument: xml file
Returns: Set with all the components of this file
'''
def naive_normalization(pascal_labels_path, xml):
    tree = ET.parse(pascal_labels_path+xml)
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
Argument: List of xml files
Does: A classes.txt file with all components of all xml_files
'''
def normalization(pascal_labels_path, xml_files):
    # create a empty set
    components = set()
    for xml in xml_files:
        components = components.union(naive_normalization(pascal_labels_path, xml))
    
    # print(components)
    components = list(components)
    components.sort()

    classes = open("classes.txt", "w")
    for component in components:
        classes.write(component+"\n")

    classes.close()
