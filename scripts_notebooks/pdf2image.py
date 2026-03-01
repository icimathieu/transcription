import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scripts_notebooks.pdf2image import convert_from_path



#fonctions : 
def all_files_in_repo(repo_path): 
    from glob import glob
    files_name = []
    for file in glob(repo_path+"*"):
        files_name.append(file)
    return(files_name)

def pdf_2_jpegs(files_list): 
    import glob
    for file in tqdm(files_list) :
        path = file+"/*.pdf"
        pdf = glob(path) 
        print(pdf)
        convert_from_path(pdf[0],
            dpi=200,
            output_folder=file, 
            poppler_path="/opt/homebrew/bin",
            fmt="jpeg",
            output_file="page")
        
def nettoyer_jpegs(file): #nettoyer mes pdfs en enlevant premières pages pdf ajoutées par la bnf


def jpegs_2_jsonls (file):
    #avec ppstructure

def jsonls_2_json (file):#ouvrir mes différents jsonl et en faire un seul json
    jsonls_path = file+"/*.jsonl"
    jsonl_files = glob.glob(jsonls_path)
    records = []
    for path in jsonl_files:
        with open(path, "r") as f:
            for line in f:
                records.append(json.loads(line))

    with open("merged_ocr.json", "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Fusionné {len(records)} entrées depuis {len(jsonl_files)} fichiers.")

#supprimer les jpg dans mon répertoire



#exécution : 

if __name__ == "__main__":
    print('oui')

    #interagir avec argparse






