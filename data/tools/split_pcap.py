import os
import gzip
from scapy.all import *


def save_first_n_packets(input_pcap, n=100):

    if input_pcap.endswith('.gz'):
        reader = PcapReader(gzip.open(input_pcap, 'rb'))
    else:
        reader = PcapReader(input_pcap)

    packets = []
    packet_count = 0

    while packet_count < n:
        packet = reader.read_packet()
        if packet is None:
            break
        packets.append(packet)
        packet_count += 1

    reader.close()

    output_file = f"split_{packet_limit}_{os.path.splitext(os.path.basename(input_pcap))[0]}.train"
    output_file = os.path.join(os.path.dirname(input_pcap), output_file)
    output_pcap = output_file.replace(".train", "") + ".train.gz"

    wrpcap(output_pcap, packets, gz=True)


# Usage example
input_file = '../train/example_train/normal.train.gz'
packet_limit = 1000

# creates a /train/example_train/split_{packet_limit}_normal.train.gz
save_first_n_packets(input_file, packet_limit)
