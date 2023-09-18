"""takes a train file and encodes the data for training"""
import gzip
from decimal import Decimal
import time
from operator import itemgetter
from itertools import tee

from data.enc import Encoder  # TODO This path is only viable, when called by a .py outside of the folder

from scapy.all import *
import scapy.contrib.modbus


def read_pcap_gz(file_path):
    with gzip.open(file_path, 'rb') as file:
        reader = PcapReader(file)

        # Loop through each packet in the reader
        for packet in reader:
            if packet.haslayer('ModbusADU'):
                ip = packet['IP']
                mb = packet.getlayer("ModbusADU")
                try:
                    f_code = mb.funcCode
                except Exception as e:
                    f_code = -1
                yield (ip.src, ip.dst, f_code), float(packet.time)


def split_to_lists(packets):
    list_of_pkt, list_of_timestamps = tee(packets)
    list_of_pkt, list_of_timestamps = [x[0] for x in list_of_pkt], [x[1] for x in list_of_timestamps]
    return list_of_pkt, list_of_timestamps


if __name__ == "__main__":
    project = 'dev_ids'
    # Provide the path to your .train.gz file
    train_data = 'data/train/normal.pcap.gz'
    # processed data, models and output will be saved in {out_dir}/{project}
    out_dir = 'out/'

    # Set the necessary parameters for testing
    # True, False
    combine_pkt_time = True
    # ['relative_to_fixed', 'relative_to_syntactic_predecessor', 'relative_to_semantic_predecessor']
    relativize_timestamp_method = 'relative_to_syntactic_predecessor'
    # ['time_feature', 'bin_uniform', 'bin_quantile', 'bin_kmeans']
    discretize_timestamp_method = 'bin_uniform'
    # ['hour', 'minute', 'second', 'microsecond'] or {'n_bins': 20}
    discretize_timestamp_parameter = {'n_bins': 20}
    timestamp_mapping = False

    # overide params from CLI
    exec(open('configurator.py').read())

    enc = Encoder(combine_pkt_time=combine_pkt_time, relativize_timestamp_method=relativize_timestamp_method,
                  discretize_timestamp_method=discretize_timestamp_method,
                  discretize_timestamp_parameter=discretize_timestamp_parameter, timestamp_mapping=timestamp_mapping)
    print(f"starting encoding of file: {train_data}")
    print(f"use encoder config: {enc.get_state()}")

    # Call the function to read the train.gz file
    packets = read_pcap_gz(train_data)
    list_of_pkt, list_of_timestamps = split_to_lists(packets)

    # Call the preprocessing_encoder method to get the encoded data
    encoded_data = enc.preprocessing_encoder(list_of_pkt, list_of_timestamps)
    print("encoded_data:", encoded_data)

    out_dir = out_dir + '/' + project
    print(f"saved at {out_dir}")

    enc.save_train_data(out_dir, encoded_data)
