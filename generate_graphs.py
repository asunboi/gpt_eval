import os
import json
from openai import OpenAI
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

def safe_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return 0
        
few_shot_path = '/gpfs/home/asun/su_lab/no_wiki_text/'
zero_shot_path = '/gpfs/home/asun/su_lab/no_follow/'
few_shot_results_path = '/gpfs/home/asun/su_lab/few_shot_output/'
zero_shot_results_path = '/gpfs/home/asun/su_lab/zero_shot_output/'


with open('/gpfs/home/asun/su_lab/indication_paths.json', 'r') as file:
    ground_data = json.load(file)

extract_unique_nodes(ground_data)

heatmap_matrix = []
heatmap_comp_matrix = []

current_index = 0
for filename in os.listdir(few_shot_path):
    if filename.endswith('.json'):
        """
        if filename != "clarithromycin__infection_due_to_staphylococcus_aureus.json":
                continue
        """
        
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

        if zero_recall > zero_gpt_recall:
            print(filename)
            
        heatmap_entry = [zero_recall,zero_precision,zero_gpt_recall,zero_gpt_precision,few_recall,few_precision,few_gpt_recall,few_gpt_precision]
        heatmap_comp_entry = [zero_gpt_recall, few_gpt_recall]
        heatmap_matrix.append(heatmap_entry)
        heatmap_comp_matrix.append(heatmap_comp_entry)

# Create heatmap
sns.heatmap(heatmap_matrix, annot=False, cmap="YlGnBu", yticklabels=False)

x_labels = ["zero_recall","zero_precision","zero_gpt_recall","zero_gpt_precision",
            "few_recall","few_precision","few_gpt_recall","few_gpt_precision"]
plt.xticks(ticks=np.arange(len(x_labels)) + 0.5, labels=x_labels, rotation=90)

plt.title("Heatmap")
plt.show()

# Create Cumulative Line Graphs
zero_gpt_recall = []
few_gpt_recall = []

for entry in heatmap_comp_matrix:
    zero_gpt_recall.append(entry[0])
    few_gpt_recall.append(entry[1])

cumulative_zero_gpt_recall = np.cumsum(zero_gpt_recall)
cumulative_few_gpt_recall = np.cumsum(few_gpt_recall)

plt.plot(cumulative_zero_gpt_recall, label='Cumulative Zero Shot GPT Recall')
plt.plot(cumulative_few_gpt_recall, label='Cumulative Few Shot GPT Recall')

plt.xlabel('Index')
plt.ylabel('Cumulative Sum')
plt.title('Zero Shot vs Few Shot Cumulative Recall')

plt.legend()
plt.show()
