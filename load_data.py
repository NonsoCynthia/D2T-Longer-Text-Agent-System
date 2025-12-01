import json
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

def extract_modified_triplesets_from_file(path):
    """
    Read XML from a file path and extract modified triplesets.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    all_triplesets = []

    for entry in root.findall("./entries/entry"):
        tripleset = []
        for mtriple in entry.findall("./modifiedtripleset/mtriple"):
            if mtriple.text is None:
                continue
            parts = [p.strip() for p in mtriple.text.split("|")]
            if len(parts) != 3:
                continue
            subj, rel, obj = parts
            tripleset.append([subj, rel, obj])  # Changed to list for consistency

        if tripleset:
            all_triplesets.append(tripleset)

    return all_triplesets


def save_result_to_json(state: dict, dataset_folder= "", filename: str = "result.json", directory: str = "results") -> None:
    """
    Saves the given agent workflow state to a JSON file in a specified directory.
    """
    # Ensure full directory path exists
    if dataset_folder != "":
        full_directory = os.path.join(directory, dataset_folder)
    else:
        full_directory = directory

    os.makedirs(full_directory, exist_ok=True)

    file_path = os.path.join(full_directory, filename)

    if os.path.isdir(file_path):
        raise IsADirectoryError(f"Cannot write to '{file_path}' because it is a directory.")

    def make_serializable(obj):
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        elif hasattr(obj, "model_dump"):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj

    serializable_state = make_serializable(dict(state))

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, indent=4)

    print(f"[SAVED] Agent result saved to: {file_path}")