from collections import Counter, defaultdict
from tqdm.auto import tqdm
from datetime import datetime
import json, re, ast
import pandas as pd
import numpy as np

class OutputConverter:
    def __init__(self):
        # self.text_dir = text_dir
        # self.text = text
        pass

    def read_text(self, text_dir = None, text = None):
        if text_dir is None and text is None:
            raise ImportError("No text file or text string is provided.")
        elif text is None:
            with open(text_dir, 'r') as f:
                text = f.readlines()
            if "{" == text[0].strip()[0]:
                self.result = text[0]
            else:
                self.result = "{"+text[0]
        else:
            self.result = "{"+text
        return self.result
    def fix_dict(self):
        corrected_str = re.sub(r"(?<=[:{,])\s*'([^']*)'\s*(?=[:,}])", r'"\1"', self.result)
        corrected_str = corrected_str.replace("'s","s")
        #corrected_str = corrected_str.replace('"s',"s")#.replace("s'","s")
        corrected_str = re.sub(r'(?<!\\)\'', '"', corrected_str)
        before, after = corrected_str.split("phenotypes")
        after = after[after.find('{')+1:].replace("}", "").strip()
        after_split = after.split(",")
        after_split_colon = [t.count(":") for t in after_split]
        phenotype_collected = []
        for i, colon_count in enumerate(after_split_colon):
            if colon_count < 2 and after_split[i].count('"') == 2:
                    pass
            else:
                phenotype_collected.append(after_split[i])    
        phenotype_str = ",".join(phenotype_collected)
        if before[-1].strip() == '"':
            corrected_str = before.strip() + 'phenotypes": {' + phenotype_str + "}}"
        elif before[-1].strip() == "'":
            corrected_str = before.strip() + "phenotypes': {" + phenotype_str + "}}"
        else:
            corrected_str = before.strip() + '"phenotypes": {' + phenotype_str + "}}"
        corrected_str = corrected_str.replace('":":','":')
        return corrected_str