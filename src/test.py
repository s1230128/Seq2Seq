import xml.etree.ElementTree as ET

data = ET.parse('../data/ted_en.xml')
root = data.getroot()

for d in root[:10]:
    for i in d:
        print(i.text)
