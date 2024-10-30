import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os, sys, re, torch, json, glob, argparse, gc, ast
from itertools import chain
from datasets import load_dataset
from tokenizers import AddedToken, pre_tokenizers
import numpy as np
from tqdm.auto import tqdm
from OutputConverter import OutputConverter
gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser(description="PhenoGPT2 Phenotypic Term Detector",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", required = True, help="directory to input folder")
parser.add_argument("-o", "--output", required = True, help="directory to output folder")
parser.add_argument("-model_dir", "--model_dir", required = False, help="directory to model folder")
args = parser.parse_args()
#Tokenizer
tokenizer_id = "Llama3_1/Meta-Llama-3.1-8B-Instruct" # Replace your tokenizer llama directory here
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

def generate_prompt(data_point):
    instruction = "You are a genetic counselor specializing in extracting demographic details and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. Your task is to provide accurate and concise information without generating random answers. When demographic details or phenotype information is not explicitly mentioned in the input, use 'unknown' as the value."
    question = "Read the following input text and generate a JSON-formatted output with the following keys: demographics and phenotypes.For the demographics key, create a sub-dictionary with age, sex, ethnicity, and race as keys, and where applicable, imply the race from ethnicity or ethnicity from race. For the phenotype key, create a sub-dictionary where each HPO term is a key, and its corresponding HPO identifier is the value. If any information is unavailable, return 'unknown' for that field.\nInput: "
    base_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|>
    
    <|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|>
    
    <|start_header_id|>assistant<|end_header_id|>
    |==|Response|==|
    {{"""
    #{model_answer}<|eot_id|><|end_of_text|>"""
    prompt = base_prompt.format(system_prompt = instruction,
                                user_prompt = question + data_point.replace("'","").replace('"',''),
                                )

    return prompt
def generate_output(model, data_point):
    prompt = generate_prompt(data_point)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    #model.to(device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_new_tokens=11000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.5, #higher temperature => generate more creative answers but the responses may change
            top_p=0.8, #higher top_p => less likely to generate random answer not in the text
    )

    response = generation_output[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True).strip()
    if len(input_ids[0]) > 11000:
        print("WARNING: Your text input has more than the predefined maximum 11000 tokens. The results may be defective.")
    return(output)
def read_text(input_file):
    if os.path.isfile(input_file):
        input_list=[input_file]
    else:
        input_list = glob.glob(input_file + "/*")
    input_dict = {}
    for f in input_list:
        file_name = f.split('/')[-1]#[:-4]
        file_name = file_name.split('.')[0]
        with open(f, 'r') as r:
            data = r.readlines()
        data = [d.strip() for d in data]
        if isinstance(data, str):
            pass
        elif len(data) > 1:
            data = "\t".join(data)
        else: # list of 1 string
            data = data[0]
        input_dict[file_name] = data
    return(input_dict)
def main():
    ##set up model
    #Model
    if args.model_dir:
        model_id = args.model_dir
    else:
        model_id = os.getcwd() + '/model/'
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "left"
    ## adding HPO IDs as new tokens so that models can treat them as a single-token.
    with open('hpo_added_tokens.json', 'r') as f:
        name2hpo = json.load(f)
    all_hpo_ids = list(np.unique(list(name2hpo.values())))
    for hpo_id in tqdm(all_hpo_ids, desc = 'Adding Tokens'):
        #tokenizer.add_tokens(AddedToken(hpo_id, normalized=False,special=False))
        tokenizer.add_tokens([AddedToken(hpo_id, single_word=True, normalized=False,special=False)], special_tokens=False)
    # #average_embedding = torch.mean(model.get_input_embeddings().weight, axis=0)

    model.resize_token_embeddings(len(tokenizer)) ## go along with tokenizer.pad_token is None
    model.config.pad_token_id = tokenizer.pad_token_id # setting pad token id for model
    model.eval()
    print('start phenogpt2')
    input_dict = read_text(args.input)
    for file_name, text in tqdm(input_dict.items()):
        try:
            print(file_name)
            # generate raw response
            raw_output = ''
            attempt = 0
            while len(raw_output.strip()) == 0 and attempt < 3:
                raw_output = generate_output(model, text)
                attempt += 1
            # clean up response
            # save output
            os.makedirs(args.output, exist_ok=True)
            output_name = args.output+"/"+file_name+"_phenogpt2.txt"
            with open(output_name, 'w') as f:
                f.write("{" + raw_output)
            # convert text strings to JSON format for future use: ### NOTE THAT: Due to the nature of LLMs, JSON-format may be problematic. Please inspect if the error comes up
            try:
                outputconverter = OutputConverter()
                raw_output = outputconverter.read_text(text = raw_output)
                processed_output = outputconverter.fix_dict()
                clean_output = ast.literal_eval(processed_output)
                with open(args.output+"/"+file_name+"_phenogpt2.json", 'w') as f:
                    json.dump(clean_output, f)
                print(clean_output)
            except:
                err = f"Please review the output file at {output_name}. The result was successfully generated; however, there may be some extra single or double quotes, or colons, which could cause a JSON format error. Please inspect and remove any of them."
                print(err)
                with open(args.output+"/"+file_name+"_phenogpt2_jsonERROR.txt", 'w') as f:
                    f.write(err)
                print(raw_output)
        except Exception as e:
            print(e)
            print("Cannot produce results for " + file_name)
if __name__ == "__main__":
    main()