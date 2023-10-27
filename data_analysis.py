import os
import glob
import tqdm
import json

unique_entity_labels = {}
unique_entity_labels = {"NotEntity": 0}
if __name__ == "__main__":
    all_annotations_path = glob.glob("./SocialHistoryMTSamples/*.ann")
    unique_key_counter = 1
    for annotation_path in tqdm.tqdm(all_annotations_path):
        with open(annotation_path, 'r') as f:
            annotation = f.readlines()
        
        for line in annotation:
            if ":" not in line:
                # print(line)
                info = line.split("\t")
                entity_type = info[1]
                entity_type = entity_type.split(" ")
                entity_type = entity_type[0]
                
                if entity_type not in unique_entity_labels:
                    unique_entity_labels[entity_type] = unique_key_counter
                    unique_key_counter += 1
                    
    with open("entity_in_dataset.json", "w") as json_file:
        json.dump(unique_entity_labels, json_file, indent=True)