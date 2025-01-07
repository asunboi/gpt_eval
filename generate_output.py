import os
import json
from openai import OpenAI
import pandas as pd
from io import StringIO

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
def generate_metrics(matching_indications, gen_data):
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
        #print("Indication Matches: " + str(ind_matches))
        #print(f"{ind_match_count}/{ind_total_count} Matched Nodes / Total Grounded Nodes")
    return ind_matches, ind_match_count, ind_total_count

#TODO 
def generate_metrics_secondary(gen_data, matching_indications):    
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
        #print("KG Matches: " + str(matches))
        #print(f"{match_count}/{total_count} Matched Nodes / Total Generated Nodes")
    return matches, match_count, total_count

prompt = """
Imagine you are my assistant tasked with comparing graphs formatted in JSON for comprehensive data parsing and matching tasks. 
The first entry provided is a knowledge graph with subheaders ['questions', , 'answers', 'triples', 'entities', 'mechanism_paragraphs]. This is a generated knowledge graph. 
Any entry given after this will be a grounded truth graph with the subheaders ['directed', 'graph', 'links', 'multigraph', 'nodes', 'reference']. 
Compare the first generated knowledge graph with all grounded truth graphs in the below manner: 
1. Compare 'entities' in the knowledge graph with 'nodes' in the grounded truth graphs. Use the examples below as reference for what entities and nodes are considered matching. 
Format the output exactly like the EXAMPLE X OUTPUT below, a tab seperated csv. Do not add any extra characters such as ``` to the output. There may only be one \t per line.
 
EXAMPLE 1 KNOWLEDGE GRAPH ENTITIES:
"entities": ["", "Procaine benzylpenicillin", "Rat-bite fever", "Streptobacillus moniliformis", "Spirillum minus", "Penicillin-binding proteins (PBPs)", "Peptidoglycan", "Bacterial cell wall", "Procaine", "Benzylpenicillin"]

EXAMPLE 1 GROUNDED TRUTH GRAPH NODES:
"nodes": [
  {
    "alt_name": "Penicillin G Procaine",
    "id": "MESH:D010402",
    "label": "Drug",
    "name": "Procaine benzylpenicillin"
  },
  {
    "id": "DB:DB01053",
    "label": "Drug",
    "name": "Benzylpenicillin"
  },
  {
    "id": "Pfam:PF00905",
    "label": "GeneFamily",
    "name": "Penicillin binding protein transpeptidase domain"
  },
  {
    "id": "GO:0009252",
    "label": "BiologicalProcess",
    "name": "peptidoglycan biosynthetic process"
  },
  {
    "id": "GO:0009273",
    "label": "BiologicalProcess",
    "name": "peptidoglycan-based cell wall biogenesis"
  },
  {
    "id": "taxonomy:34105",
    "label": "OrganismTaxon",
    "name": "Streptobacillus moniliformis"
  },
  {
    "id": "MESH:D011906",
    "label": "Disease",
    "name": "Rat-Bite Fever"
  }
]

EXAMPLE 1 OUTPUT:
Entities\tGrounded
Procaine benzylpenicillin\tProcaine benzylpenicillin
Rat-bite fever\tRat-bite fever
Streptobacillus moniliformis\tStreptobacillus moniliformis
Spirillum minus\t
Penicillin-binding proteins (PBPs)\tPenicillin binding protein transpeptidase domain
Peptidoglycan\tpeptidoglycan biosynthetic process
Bacterial cell wall\tpeptidoglycan-based cell wall biogenesis
Benzylpenicillin\tBenzylpenicillin

EXAMPLE 2 KNOWLEDGE GRAPH ENTITIES:
"entities": ["", "Betamethasone", "Corticosteroid", "Trichinellosis", "Trichinella larvae", "Cortisol", "Adrenal glands", "Glucocorticoid receptors", "Prostaglandins", "Leukotrienes", "Muscles"]

EXAMPLE 2 GROUNDED TRUTH GRAPH NODES:
"nodes": [
  {
    "id": "MESH:D001623",
    "label": "Drug",
    "name": "betamethasone"
  },
  {
    "id": "UniProt:P04150",
    "label": "Protein",
    "name": "Glucocorticoid receptor"
  },
  {
    "id": "UniProt:P04083",
    "label": "Protein",
    "name": "Annexin A1"
  },
  {
    "id": "UniProt:P47712",
    "label": "Protein",
    "name": "Cytosolic phospholipase A2"
  },
  {
    "id": "UniProt:P23219",
    "label": "Protein",
    "name": "Prostaglandin G/H synthase 1"
  },
  {
    "id": "UniProt:P35354",
    "label": "Protein",
    "name": "Prostaglandin G/H synthase 2"
  },
  {
    "id": "GO:0019370",
    "label": "BiologicalProcess",
    "name": "leukotriene biosynthetic process"
  },
  {
    "id": "GO:0001516",
    "label": "BiologicalProcess",
    "name": "prostaglandin biosynthetic process"
  },
  {
    "id": "GO:0006954",
    "label": "BiologicalProcess",
    "name": "inflammatory response"
  },
  {
    "id": "taxonomy:6333",
    "label": "OrganismTaxon",
    "name": "Trichinella"
  },
  {
    "id": "MESH:D014235",
    "label": "Disease",
    "name": "Trichinellosis"
  }
]

EXAMPLE 2 OUTPUT:
Entities\tGrounded
Betamethasone\tbetamethasone
Corticosteroid\t
Trichinellosis\tTrichinellosis
Trichinella larvae\tTrichinella
Cortisol\t
Adrenal glands\t
Glucocorticoid receptors\tGlucocorticoid receptor
Prostaglandins\tProstaglandin G/H synthase 1; Prostaglandin G/H synthase 2
Leukotrienes\tleukotriene biosynthetic process
Muscles\t

EXAMPLE 3 KNOWLEDGE GRAPH ENTITIES:
"entities": ["", "Aluminum hydroxide", "Gastric ulcers", "Stomach lining", "Hydrochloric acid", "Aluminum chloride", "Water", "Helicobacter pylori", "Nonsteroidal anti-inflammatory drugs (NSAIDs)"]

EXAMPLE 3 GROUNDED TRUTH GRAPH NODES:
"nodes": [
  {
    "id": "MESH:D000536",
    "label": "Drug",
    "name": "Aluminum hydroxide"
  },
  {
    "id": "MESH:D005744",
    "label": "ChemicalSubstance",
    "name": "Gastric Acid"
  },
  {
    "id": "InterPro:IPR034162",
    "label": "GeneFamily",
    "name": "Pepsin catalytic domain"
  },
  {
    "id": "MESH:D013276",
    "label": "Disease",
    "name": "Gastric ulcer"
  }
]
    
EXAMPLE 3 OUTPUT:
Entities\tGrounded
Aluminum hydroxide\tAluminum hydroxide
Gastric ulcers\tGastric ulcer
Stomach lining\t
Hydrochloric acid\tGastric Acid
Aluminum chloride\t
Water\t
Helicobacter pylori\t
Nonsteroidal anti-inflammatory drugs (NSAIDs)\t

"""

# ADD OPENAI KEY HERE
#os.environ['OPENAI_API_KEY'] = 

few_shot_path = '/gpfs/home/asun/su_lab/no_wiki_text/'
zero_shot_path = '/gpfs/home/asun/su_lab/no_follow/'
few_shot_results = '/gpfs/home/asun/su_lab/few_shot_output'
zero_shot_results = '/gpfs/home/asun/su_lab/zero_shot_output'


with open('/gpfs/home/asun/su_lab/indication_paths.json', 'r') as file:
    ground_data = json.load(file)

extract_unique_nodes(ground_data)

for filename in os.listdir(few_shot_path):
    if filename.endswith('.json'):

        """
        if filename != "doxycycline__trachoma.json":
                continue
        """
        
        #print(prompt)
        
        with open(os.path.join(few_shot_path, filename), 'r') as file:
            few_shot_data = json.load(file)
        with open(os.path.join(zero_shot_path, filename), 'r') as file:
            zero_shot_data = json.load(file)
        print(filename)
        
        drug_disease = filename.split('.')[0]
        drug = drug_disease.split('__')[0]
        drug = drug.replace('_', ' ')
        disease = drug_disease.split('__')[1]
        disease = disease.replace('_', ' ')
        
        exists, matching_indications = extract_matching_indications(drug, disease, ground_data)            
        print("Number of Indication Pathways: " + str(len(matching_indications)))
        
        for i in range(len(matching_indications)):
            indication = matching_indications[i]
            print("Zero Shot Metrics")
            matches, match_count, total_count = generate_metrics_secondary(zero_shot_data, matching_indications)
            zero_matches, zero_match_count, zero_total_count = generate_metrics(matching_indications, zero_shot_data)
            
            content = """"""
            content += json.dumps(zero_shot_data, indent=4) + '\n' + '\n'
            content += json.dumps(indication, indent=4) + '\n'
    
            client = OpenAI()
            
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )

            # TEST CHATGPT OUTPUT
            #print(completion.choices[0].message.content)
    
            with open(f"{zero_shot_results}/{drug_disease}_{i}.csv", "w") as file:
                file.write(completion.choices[0].message.content)
            
            print("Few Shot Metrics")
            matches, match_count, total_count = generate_metrics_secondary(few_shot_data, matching_indications)
            few_shot_matches, few_shot_match_count, few_shot_total_count = generate_metrics(matching_indications, few_shot_data)
            
            content = """"""
            content += json.dumps(few_shot_data, indent=4) + '\n' + '\n'
            content += json.dumps(indication, indent=4) + '\n'
    
            client = OpenAI()
    
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
    
            print(completion.choices[0].message.content)
    
            with open(f"{few_shot_data}/{drug_disease}_{i}.csv", "w") as file:
                file.write(completion.choices[0].message.content)
                
            print("\n")
