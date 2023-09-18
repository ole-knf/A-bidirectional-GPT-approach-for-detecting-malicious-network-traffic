import gzip
import json
import os
attack1 = [
    (1617992748.0, 1617992894.0, 'MITM'),
    (1617993008.0, 1617993074.0, 'physical fault'), 
    (1617993251.0, 1617993338.0, 'MITM'),
    (1617993518.0, 1617993590.0, 'physical fault'),
    (1617993832.0, 1617993954.0, 'MITM'),
    (1617994142.0, 1617994278.0, 'physical fault'),
    (1617994685.0, 1617994772.0, 'MITM'),
    (1617994840.0, 1617994927.0, 'MITM')
    ]

attack2 = [
    (1618846732.0, 1618846732.0, 'Scan'),
    (1618846809.0, 1618846809.0, 'Scan'),
    (1618846870.0, 1618846870.0, 'Scan'),
    (1618846923.0, 1618846923.0, 'Scan'),
    (1618847026.0, 1618847062.0, 'DoS'),
    (1618847295.0, 1618847336.0, 'physical fault'),
    (1618847476.0, 1618847476.0, 'Scan'),
    (1618847679.0, 1618847759.0, 'physical fault'),
    (1618847920.0, 1618847920.0, 'Scan'),
    (1618847997.0, 1618848010.0, 'DoS'),
    (1618848279.0, 1618848279.0, 'Scan'),
    (1618848506.0, 1618848601.0, 'MITM'),
    (1618848706.0, 1618848734.0, 'DoS')
    ]

attack3 = [
    (1617997397.0, 1617997448.0, 'physical fault'),
    (1617997552.0, 1617997581.0, 'DoS'),
    (1617997794.0, 1617997848.0, 'physical fault'),
    (1617998040.0, 1617998085.0, 'physical fault'),
    (1617998129.0, 1617998175.0, 'DoS'),
    (1617998282.0, 1617998349.0, 'MITM'),
    (1617998478.0, 1617998523.0, 'MITM')
]

attack4 = [
    (1645454854.0, 1645454854.0, 'Scan'),
    (1645454897.0, 1645454897.0, 'Scan'),
    (1645454925.0, 1645454925.0, 'Scan'),
    (1645454966.0, 1645454966.0, 'Scan'),
    (1645455094.0, 1645455094.0, 'Scan'),
    (1645455138.0, 1645455179.0, 'DoS'),
    (1645455346.0, 1645455383.0, 'physical fault'),
    (1645455446.0, 1645455466.0, 'DoS'),
    (1645455504.0, 1645455559.0, 'physical fault'),
    (1645455716.0, 1645455754.0, 'physical fault'),
    (1645455851.0, 1645456051.0, 'MITM'),
    (1645456147.0, 1645456147.0, 'Scan'),
    (1645456166.0, 1645456166.0, 'Scan'),
    (1645456345.0, 1645456435.0, 'DoS'),
]
attacklist = [attack1, attack2, attack3]
# Specify the path to your compressed .gz file
# TODO add support for new attack files generation
file_path = 'data/attack_ipal/bin_kmeans'
file_name = 'GPT_attack_1_split_2.pcap.ipal.gz' 
file = os.path.join(file_path, file_name)
# Create a copy of the decompressed data
output_data = []

# Open the compressed .gz file and decompress it while reading
with gzip.open(file, 'rt') as file:
    for line in file:
        # Parse each line as JSON
        entry = json.loads(line)
        
        # Modify the "malicious" field based on a condition
        timestamp = float(entry["timestamp"])
        
        for attack in attacklist:
            for interval in attack:
                if (timestamp >= (interval[0] - 7200)) and (timestamp <= (interval[1] - 7200)):
                    entry["malicious"] = interval[2]
        
        for interval in attack4:
            if (timestamp >= (interval[0] - 3600)) and (timestamp <= (interval[1] - 3600)):
                entry["malicious"] = interval[2]
        
        # Append the modified entry to the output data
        output_data.append(entry)

# Specify a new file where you want to save the modified data
output_path = file_path+'_rewritten'
output_file = os.path.join(output_path, file_name)

# Compress and write the modified data back to a .gz file
with gzip.open(output_file, 'wt') as file:
    for entry in output_data:
        # Convert the modified entry back to JSON format
        json_entry = json.dumps(entry)
        
        # Write the JSON entry to the output file
        file.write(json_entry + '\n')

# The modified data is now saved in 'output_file.gz'

