from collections import defaultdict
import xml.etree.ElementTree as ET

def extract_mtriples_by_category(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    category_dict = defaultdict(list)

    for entry in root.findall(".//entry"):
        category = entry.attrib.get("category", "Unknown")
        mtriples = []
        
        for mtriple in entry.findall(".//mtriple"):
            parts = mtriple.text.strip().split(" | ")
            if len(parts) == 3:
                mtriples.append(parts)
        
        category_dict[category].append(mtriples)

    # Optionally sort the categories alphabetically
    sorted_category_dict = dict(sorted(category_dict.items()))
    
    return sorted_category_dict

def extract_mtriples(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    all_mtriples = []
    
    # Iterate through each entry in the XML
    for entry in root.findall(".//entry"):
        mtriples = []
        for mtriple in entry.findall(".//mtriple"):
            parts = mtriple.text.split(" | ")
            if len(parts) == 3:
                mtriples.append(parts)
        all_mtriples.append(mtriples)
    
    return all_mtriples