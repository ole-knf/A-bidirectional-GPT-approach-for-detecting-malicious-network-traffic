import gzip
import os

def gzip_encode(input_file, output_file):
    with open(input_file, 'rb') as f_in, gzip.open(output_file, 'wb') as f_out:
        f_out.writelines(f_in)

# Usage example
input_file = '../attack/attack_4.pcap'

output_dir = os.path.dirname(input_file)
output_file = f"{os.path.splitext(os.path.basename(input_file))[0]}.pcap.gz"

gzip_encode(input_file, os.path.join(output_dir, output_file))
