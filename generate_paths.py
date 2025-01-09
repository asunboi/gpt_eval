import os
import json
from openai import OpenAI
import pandas as pd
from io import StringIO
from collections import defaultdict

def extract_unique_nodes(ground_data):
    unique_id = {}
    unique_name = {}

    for entry in ground_data:
        for node in entry['nodes']:
            node_id = node['id']
            if node_id not in unique_id:
                unique_id[node_id] = {
                    'label': node['label'],
                    'name': node['name']
                }
            node_name = node['name'].lower()
            if node_name not in unique_name:
                unique_name[node_name] = {
                    'label': node['label'],
                    'id': node['id']
                }

    return unique_id, unique_name


def extract_matching_indications(drug, disease, ground_data):
    # Check if KG pathway exists in indications.json and append matching entries
    exists = False
    matching_indications = []

    for entry in ground_data:
        if entry['graph']['disease'].lower() == disease.lower() and entry['graph'][
            'drug'].lower() == drug.lower():
            exists = True
            matching_indications.append(entry)

    return exists, matching_indications

#TODO output metrics for all entries in matching indications.
def get_hc_metrics(matching_indications, gen_data):
    for i in range(len(matching_indications)):
        entry = matching_indications[i]
        ind_matches = [0] * len(matching_indications[i]['nodes'])
        ind_match_count = 0
        ind_total_count = len(matching_indications[i]['nodes'])
        current_index = 0
        for node in entry['nodes']:
            for value in gen_data['entities']:
                if node['name'].lower() == value.lower():
                    ind_matches[current_index] = 1
                    ind_match_count += 1
                    break
            current_index += 1
    for entry in matching_indications:
        matches = [0] * len(gen_data['entities'])
        match_count = 0
        total_count = len(gen_data['entities'])
        current_index = 0
        for value in gen_data['entities']:
            for node in entry['nodes']:
                if node['name'].lower() == value.lower():
                    matches[current_index] = 1
                    match_count += 1
                    #print(value.lower())
                    break
            current_index += 1
    precision = match_count/total_count
    recall = ind_match_count/ind_total_count
    return precision, recall
    
def get_gpt_metrics(results, indication):
    df = pd.read_csv(StringIO(results), sep='\t', usecols=[0, 1])
    count = ((df['Entities'].notnull()) & (df['Grounded'].notnull())).sum()
    gen_nodes = df['Entities'].dropna().str.split(';').explode().tolist()
    # Split the strings by ";" and flatten into a single list
    all_nodes = df['Grounded'].dropna().str.split(';').explode().tolist()
    # Get unique values
    unique_nodes = list(set(all_nodes))
    total_nodes = 0
    current_nodes = 0
    for node in indication['nodes']:
        total_nodes += 1
        for value in unique_nodes:
            if node['name'] == value:
                current_nodes += 1
                break
    recall = current_nodes/total_nodes  
    precision = count / len(gen_nodes)
    return precision, recall

def make_dictionary(results, indication):
    df = pd.read_csv(StringIO(results), sep='\t', usecols=[0, 1])
    count = ((df['Entities'].notnull()) & (df['Grounded'].notnull())).sum()
    gen_nodes = df['Entities'].dropna().str.split(';').explode().tolist()
    # Split the strings by ";" and flatten into a single list
    all_nodes = df['Grounded'].dropna().str.split(';').explode().tolist()
    # Get unique values
    unique_nodes = list(set(all_nodes))
    total_nodes = 0
    current_nodes = 0
    for node in indication['nodes']:
        total_nodes += 1
        for value in unique_nodes:
            if node['name'] == value:
                current_nodes += 1
                break
    """
    result_dict = {
    row["Grounded"]: row["Entities"]
    for _, row in df.dropna(subset=["Entities", "Grounded"]).iterrows()
    }
    #print(result_dict)
    """
    result_dict = defaultdict(list)  # Initialize a defaultdict with list as the default factory
    
    for _, row in df.dropna(subset=["Entities", "Grounded"]).iterrows():
        result_dict[row["Grounded"]].append(row["Entities"])
    
    # Convert defaultdict back to a regular dictionary if needed
    result_dict = dict(result_dict)
    return result_dict

def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
        
few_shot_path = '/gpfs/home/asun/su_lab/no_wiki_text/'
zero_shot_path = '/gpfs/home/asun/su_lab/no_follow/'
few_shot_results_path = '/gpfs/home/asun/su_lab/few_shot_output/'
zero_shot_results_path = '/gpfs/home/asun/su_lab/zero_shot_output/'

import os
import json

def process_file(few_shot_path, filename, indication, result_dict, output_path, iteration):
    """
    Process a file to extract triples, map IDs to names, and write results to a CSV file.

    Args:
        few_shot_path (str): Path to the directory containing the file.
        filename (str): Name of the file to process.
        indication (dict): Dictionary containing 'nodes' and 'links'.
        result_dict (dict): Dictionary containing node match results.
        output_path (str): Path to save the output CSV file.
        iteration (int): Iteration number for naming the output file.
    """
    print(filename)
    
    # Load the few-shot data
    with open(os.path.join(few_shot_path, filename), 'r') as file:
        few_shot_data = json.load(file)
    
    # Parse triples into a dictionary
    path_dictionary = {}
    for triple in few_shot_data['triples']:
        temp_triple = triple.split(" -> ")
        if len(temp_triple) == 3:
            key = temp_triple[0]
            value = (temp_triple[1], temp_triple[2])
            if key not in path_dictionary:
                path_dictionary[key] = [value]
            else:
                path_dictionary[key].append(value)
    
    # Extract drug and disease names
    drug_disease = filename.split('.')[0]
    drug = drug_disease.split('__')[0].replace('_', ' ')
    disease = drug_disease.split('__')[1].replace('_', ' ')
    
    # Create a dictionary mapping node IDs to names
    id_to_name = {node['id']: node['name'] for node in indication['nodes']}
    
    # Generate the list of tuples
    links_with_names = [
        (id_to_name[link['source']], link['key'], id_to_name[link['target']])
        for link in indication['links']
    ]
    
    # Match links with paths
    path_dict = {}
    for link in links_with_names:
        path_dict[link] = []
        for match in result_dict[link[0]]:
            for value in path_dictionary.get(match, []):
                for second_match in result_dict[link[2]]:
                    if value[1] == second_match:
                        path_dict[link].append((match, value[0], value[1]))
    
    # Write results to a CSV file
    output_file = os.path.join(output_path, f"{drug_disease}_{iteration}.csv")
    with open(output_file, "w") as file:
        for key, values in path_dict.items():
            if values:
                # Write the key and the first value
                file.write(f"{key}\t{values[0]}\n")
                # Write the remaining values, indented
                for value in values[1:]:
                    file.write(f"\t{value}\n")
            else:
                file.write(f"{key}\t\n")
                
def get_path(few_shot_path, filename):
    with open(os.path.join(few_shot_path, filename), 'r') as file:
        few_shot_data = json.load(file)

    path_dictionary = {}
    
    for triple in few_shot_data['triples']:
        temp_triple = triple.split(" -> ")
        key = temp_triple[0]
        value = (temp_triple[1],temp_triple[2])
        if key not in path_dictionary:
            path_dictionary[key] = [value]
        else:
            path_dictionary[key].append(value)

    drug_disease = filename.split('.')[0]
    drug = drug_disease.split('__')[0]
    drug = drug.replace('_', ' ')
    disease = drug_disease.split('__')[1]
    disease = disease.replace('_', ' ')

    #print(path_dictionary)
    exists, matching_indications = extract_matching_indications(drug, disease, ground_data)

    for indication in matching_indications:
        # Create a dictionary mapping node IDs to names
        id_to_name = {node['id']: node['name'] for node in indication['nodes']}
        
        # Generate the list of tuples
        links_with_names = [
            (id_to_name[link['source']], link['key'], id_to_name[link['target']])
            for link in indication['links']
        ]


    path_dict = {}
    #print(path_dictionary)
    for link in links_with_names:
        path_dict[link] = []
        for value in path_dictionary[result_dict[link[0]]]:
            if value[1] == result_dict[link[2]]:
                path_dict[link].append((result_dict[link[0]], value[0], value[1]))

    #print(path_dict)

    with open(f"/gpfs/home/asun/su_lab/zs_path_output/{drug_disease}.csv", "w") as file:
        for key, values in path_dict.items():
            # Write the key and the first value
            file.write(f"{key}\t{values[0]}\n")
            # Write the remaining values, indented
            for value in values[1:]:
                file.write(f"\t{value}\n")
                
with open('/gpfs/home/asun/su_lab/indication_paths.json', 'r') as file:
    ground_data = json.load(file)

extract_unique_nodes(ground_data)

heatmap_matrix = []
heatmap_comp_matrix = []

zs_perfect_recall = []
fs_perfect_recall = []

current_index = 0
for filename in os.listdir(few_shot_path):
    if filename.endswith('.json'):
        """
        if filename != "ergometrine__postpartum_hemorrhage.json":
                continue
        #"""
        
        #print(prompt)
        
        with open(os.path.join(few_shot_path, filename), 'r') as file:
            few_shot_data = json.load(file)
        with open(os.path.join(zero_shot_path, filename), 'r') as file:
            zero_shot_data = json.load(file)

        #print(filename)
        
        drug_disease = filename.split('.')[0]
        drug = drug_disease.split('__')[0]
        drug = drug.replace('_', ' ')
        disease = drug_disease.split('__')[1]
        disease = disease.replace('_', ' ')

        exists, matching_indications = extract_matching_indications(drug, disease, ground_data)
        max_few_shot_recall = 0 
        max_zero_shot_recall = 0
        max_zero_shot_precision = 0
        max_few_shot_precision = 0
        for i in range(len(matching_indications)):
            indication = matching_indications[i]
            with open(os.path.join(few_shot_results_path, f"{drug_disease}_{i}.csv"), 'r') as file:
                few_shot_results = file.read()
                current_precision, current_recall = get_gpt_metrics(few_shot_results, indication)
                #print(current_recall
                if current_precision > max_few_shot_precision:
                    max_few_shot_precision = current_precision
                if current_recall > max_few_shot_recall:
                    max_few_shot_recall = current_recall
                if current_recall == 1:
                    result_dict = make_dictionary(few_shot_results, indication)
                    process_file(
                        few_shot_path,
                        filename,
                        indication,
                        result_dict,
                        output_path="/gpfs/home/asun/su_lab/fs_path_output/",
                        iteration=i
                    )
            with open(os.path.join(zero_shot_results_path, f"{drug_disease}_{i}.csv"), 'r') as file:
                zero_shot_results = file.read()
                #print(f"{drug_disease}_{i}.csv")
                #escaped_string = zero_shot_results.replace("\t", "\\t")
                #print(escaped_string)
                current_precision, current_recall = get_gpt_metrics(zero_shot_results, indication)
                #print(current_recall)
                if current_precision > max_zero_shot_precision:
                    max_zero_shot_precision = current_precision
                if current_recall > max_zero_shot_recall:
                    max_zero_shot_recall = current_recall
                if current_recall == 1:
                    result_dict = make_dictionary(zero_shot_results, indication)
                    #print(result_dict)
                    process_file(
                        zero_shot_path,
                        filename,
                        indication,
                        result_dict,
                        output_path="/gpfs/home/asun/su_lab/zs_path_output/",
                        iteration=i
                    )
            
        few_shot_results = safe_int(few_shot_results[-1])
        zero_shot_results = safe_int(zero_shot_results[-1])

        #print(few_shot_results)

        #print("Number of Indication Pathways: " + str(len(matching_indications)))
        #print("Zero Shot Metrics")
        zero_precision, zero_recall = get_hc_metrics(matching_indications, zero_shot_data)
        zero_gpt_precision = max_zero_shot_precision
        zero_gpt_recall = max_zero_shot_recall
        
        #print("Few Shot Metrics")
        few_precision, few_recall = get_hc_metrics(matching_indications, few_shot_data)
        few_gpt_precision = max_few_shot_precision
        few_gpt_recall = max_few_shot_recall

        #if max_zero_shot_recall == 1:
            
        #if max_few_shot_recall == 1:
            
            
        heatmap_entry = [zero_recall,zero_precision,zero_gpt_recall,zero_gpt_precision,few_recall,few_precision,few_gpt_recall,few_gpt_precision]
        heatmap_comp_entry = [zero_gpt_recall, few_gpt_recall]
        heatmap_matrix.append(heatmap_entry)
        heatmap_comp_matrix.append(heatmap_comp_entry)
        