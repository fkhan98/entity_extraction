import os
import glob
import tqdm
import json
# from fuzzywuzzy import fuzz

os.system("rm -rf ./dataset2")
os.mkdir("./dataset/")
os.mkdir("./dataset/train")
os.mkdir("./dataset/valid")

with open('entity_label_map.json', 'r') as file:
    entity_label = json.load(file)
entities = list(entity_label.keys())


def generate_and_text_and_labels(text, text_path, sorted_dict, split="train"):
    text = text.rstrip()
    text = text.replace("  "," ")
    guid = text_path.split("/")[-1]

    temp_start = 0
    for value in sorted_dict.values():
        tag = value[0]
        tag_start = int(value[1])
        tag_start -= 1
        tag_end = int(value[2])
        ent = text[tag_start:tag_end]

        try:
            if ent[0] == " ":
                tag_start += 1
        except:
            continue

        not_entity_substring = text[temp_start:tag_start].rstrip()
        not_entity_substring = not_entity_substring.lstrip()
        if not_entity_substring == '':
            continue
        not_entity_words = not_entity_substring.split(" ")
        for word in not_entity_words:
            with open(f"./dataset/{split}/{guid}", "a") as f:
                f.write(f"{word}\tNotEntity\n")

        entity = text[tag_start:tag_end]
        entity = entity.rstrip()
        entity_sections = entity.split(" ")
        for i, section in enumerate(entity_sections):
            with open(f"./dataset/{split}/{guid}", "a") as f:
                if i == 0:
                    f.write(f"{section}\t{tag}-B\n")
                else:
                    f.write(f"{section}\t{tag}-I\n")
        temp_start = tag_end
    
    ## remaining string with no entities
    not_entity_substring = text[temp_start:-1].rstrip()
    not_entity_substring = not_entity_substring.lstrip()
    if not_entity_substring == '':
        return
    else:
        not_entity_words = not_entity_substring.split(" ")
        for word in not_entity_words:
            with open(f"./dataset/{split}/{guid}", "a") as f:
                f.write(f"{word}\tNotEntity\n")
    
    # exit()

if __name__ == "__main__":
    text_paths = glob.glob("./SocialHistoryMTSamples/*.txt")
    train_texts_paths = text_paths[0:int(0.8*len(text_paths))]
    valid_texts_paths = text_paths[int(0.8*len(text_paths)):]
    train_valids_splits = [train_texts_paths, valid_texts_paths]
    
    train_valids_splits = [train_texts_paths, valid_texts_paths]
    for i, split in enumerate(train_valids_splits):
        for text_path in tqdm.tqdm(split):
            annotation_path = text_path.replace(".txt",".ann")
            with open(annotation_path, "r") as ann_file:
                annotation = ann_file.readlines()
                
            ##checking for empty annotation
            if len(annotation) == 0:
                continue      
            
            with open(text_path, "r") as txt_file:
                text = txt_file.readlines()
            
            if len(text) == 2:
                temp = {}
                for line in annotation:
                    if ":" not in line:
                        entity_location = line.split("\t")
                        
                        if len(entity_location) > 2:
                            entity_type = entity_location[1].split(" ")[0]
                            if entity_type in entities:
                                # print(entity_location)
                                values = entity_location[1]
                                tags = values.split(" ")
                                # print(tags)
                                try:
                                    temp[int(tags[1])] = tags
                                except:
                                    pass
            
                sorted_items = sorted(temp.items())   
                sorted_dict = dict(sorted_items)

                ##remove overlap tokens
                end_prev = -1
                keys_to_remove = []
                for key in sorted_dict.keys():
                    start = sorted_dict[key][1]
                    if int(start) > end_prev:
                        end_prev = int(sorted_dict[key][2])
                    else:
                        end_prev = int(sorted_dict[key][2])
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    sorted_dict.pop(key)
                    
                if i == 0:
                    generate_and_text_and_labels(text[0], text_path, sorted_dict, "train")
                else:
                    # exit()
                    generate_and_text_and_labels(text[0], text_path, sorted_dict, "valid")
           