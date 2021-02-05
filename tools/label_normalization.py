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
Returns: A dictionary containing the quantity of each component
'''
def normalization_method(pascal_labels_path, xml, dic):
    tree = ET.parse(pascal_labels_path+xml)
    root = tree.getroot()

    components = []

    for child in root.findall("object"):
        components.append(child[0].text)

    # verifica se é connector Port
    # se não, pega a primeira palavra
    for i in range(len(components)):
        words = components[i].split()
        if len(words)>=2:
            if words[0]+words[1]=='connector"Port':
                print(components[i])
                components[i] = "connectorPort"
            else:
                if '"' in words[0]:
                    components[i] = words[0].split('"')[1]
                else:
                    components[i] = words[0]
        else:
            components[i] = words[0]
            
        if components[i] not in dic:
            dic[components[i]] = 1
        else:
            dic[components[i]] += 1

    return dic
    
'''
Argument: pascal labels path
Does: A classes.txt file with all components of all xml files
Returns: A dictionary containing the quantity of each component
'''
def normalization(pascal_labels_path):
    xml_files = list_xml_files(pascal_labels_path)
    # create a empty set
    components = set()
    dic = {}
    for xml in xml_files:
        dic = normalization_method(pascal_labels_path, xml, dic)
    
    classes = open("classes.txt", "w")
    for component in sorted(dic):
        classes.write(component+"\n")

    classes.close()
    return dic