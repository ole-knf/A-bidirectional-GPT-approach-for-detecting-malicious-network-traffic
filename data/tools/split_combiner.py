import gzip
import json
import os


file_path = 'data/attack_ipal/bin_kmeans_rewritten'
file_name1 = 'GPT_attack_1_split_1.pcap.ipal.gz'
file_name2 = 'GPT_attack_1_split_2.pcap.ipal.gz'
file1 = os.path.join(file_path, file_name1)
file2 = os.path.join(file_path, file_name2)

output_data = []
current_id = 0

for file in [file1,file2]:
    with gzip.open(file, 'rt') as file:
        for line in file:
            entry = json.loads(line) 
            entry["id"] = current_id
            output_data.append(entry)
            current_id += 1

output_file = os.path.join(file_path, 'GPT_attack_1.ipal.gz')

with gzip.open(output_file, 'wt') as file:
    for entry in output_data:
        json_entry = json.dumps(entry)
        file.write(json_entry + '\n')