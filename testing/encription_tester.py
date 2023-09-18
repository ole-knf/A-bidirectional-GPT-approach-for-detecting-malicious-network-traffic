import contextlib
import random
import time
from io import StringIO
import sys

from data.enc import Encoder
from data.pcap2enc import read_pcap_gz


# Function to generate a random packet
def generate_packet():
    src = str(random.randint(1, 5))
    dest = str(random.randint(1, 5))
    func_code = "func_c " + str(random.randint(1, 5))
    return (src, dest, func_code)


def test_single():
    # Set the necessary parameters for testing
    combine_pkt_time = True
    # ['relative_to_fixed', 'relative_to_syntactic_predecessor', 'relative_to_semantic_predecessor']
    relativize_timestamp_method = 'relative_to_fixed'
    discretize_timestamp_method = 'bin_uniform'
    discretize_timestamp_parameter = {'n_bins': 2}
    # discretize_timestamp_method = 'time_feature'
    # discretize_timestamp_parameter = ['hour', 'minute', 'second', 'microsecond']
    timestamp_mapping = False

    # Define a list of packets and timestamps for testing
    np = 3
    list_of_pkt = [generate_packet() for _ in range(np)]
    start_timestamp = int(time.mktime(time.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")))
    list_of_timestamps = [start_timestamp + i * 1000 for i in range(np)]

    # add identical package
    list_of_pkt.append(list_of_pkt[0])
    list_of_timestamps.append(list_of_timestamps[0])

    # add same package but 1000 later
    list_of_pkt.append(list_of_pkt[0])
    list_of_timestamps.append(list_of_timestamps[0] + 1000)

    # add other package at same time
    list_of_pkt.append(list_of_pkt[1])
    list_of_timestamps.append(list_of_timestamps[0] + 1000)

    print("list of pkts:", list_of_pkt)
    print("list of timestamps:", list_of_timestamps)

    # Create an instance of the class to test
    encoder = Encoder(combine_pkt_time=combine_pkt_time, relativize_timestamp_method=relativize_timestamp_method,
                  discretize_timestamp_method=discretize_timestamp_method,
                  discretize_timestamp_parameter=discretize_timestamp_parameter, timestamp_mapping=timestamp_mapping)

    # Call the preprocessing_encoder method to get the encoded data
    encoded_data = encoder.preprocessing_encoder(list_of_pkt, list_of_timestamps)
    print(encoder.get_state())
    print(encoded_data)

    print("Testing completed successfully.")

    decoded = encoder.decode(encoded_data[0])
    print(decoded)


def test_preprocessing_encoder():
    combi = {
        0:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': None,
                'discretize_timestamp_method': None,
                'discretize_timestamp_parameter': None,
                'timestamp_mapping': False
            },
        1:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'bin_uniform',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        2:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_syntactic_predecessor',
                'discretize_timestamp_method': 'bin_uniform',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        3:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_semantic_predecessor',
                'discretize_timestamp_method': 'bin_uniform',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        4:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'bin_quantile',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        5:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_syntactic_predecessor',
                'discretize_timestamp_method': 'bin_quantile',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        6:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_semantic_predecessor',
                'discretize_timestamp_method': 'bin_quantile',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        7:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'bin_kmeans',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        8:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_syntactic_predecessor',
                'discretize_timestamp_method': 'bin_kmeans',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        9:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_semantic_predecessor',
                'discretize_timestamp_method': 'bin_kmeans',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': False
            },
        10:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'bin_uniform',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        11:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_syntactic_predecessor',
                'discretize_timestamp_method': 'bin_uniform',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        12:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_semantic_predecessor',
                'discretize_timestamp_method': 'bin_uniform',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        13:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'bin_quantile',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        14:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_syntactic_predecessor',
                'discretize_timestamp_method': 'bin_quantile',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        15:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_semantic_predecessor',
                'discretize_timestamp_method': 'bin_quantile',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        16:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'bin_kmeans',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        17:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_syntactic_predecessor',
                'discretize_timestamp_method': 'bin_kmeans',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        18:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_semantic_predecessor',
                'discretize_timestamp_method': 'bin_kmeans',
                'discretize_timestamp_parameter': {'n_bins': 3},
                'timestamp_mapping': True
            },
        19:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'time_feature',
                'discretize_timestamp_parameter': ['hour', 'minute', 'second', 'microsecond'],
                'timestamp_mapping': True
            },
        20:
            {
                'combine_pkt_time': True,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'time_feature',
                'discretize_timestamp_parameter': ['hour', 'minute', 'second', 'microsecond'],
                'timestamp_mapping': False
            },
        21:
            {
                'combine_pkt_time': False,
                'relativize_timestamp_method': 'relative_to_fixed',
                'discretize_timestamp_method': 'time_feature',
                'discretize_timestamp_parameter': ['hour', 'minute', 'second', 'microsecond'],
                'timestamp_mapping': False
            },
    }

    # Define a list of packets and timestamps for testing
    np = 3
    list_of_pkt = [generate_packet() for _ in range(np)]
    start_timestamp = int(time.mktime(time.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")))
    list_of_timestamps = [start_timestamp + i * 1000 for i in range(np)]

    # add identical package
    list_of_pkt.append(list_of_pkt[0])
    list_of_timestamps.append(list_of_timestamps[0])

    # add same package but 1000 later
    list_of_pkt.append(list_of_pkt[0])
    list_of_timestamps.append(list_of_timestamps[0] + 1000)

    # add other package at same time
    list_of_pkt.append(list_of_pkt[1])
    list_of_timestamps.append(list_of_timestamps[0] + 1000)

    print("checking with list_of_pkt:", list_of_pkt)
    print("and list_of_timestamps:", list_of_timestamps)

    # Loop through each combination
    for combi_id, params in combi.items():
        print(f"testing {combi_id} with {params}")
        # Create an instance of the class with the combination parameters
        enc = Encoder(
            combine_pkt_time=params['combine_pkt_time'],
            relativize_timestamp_method=params['relativize_timestamp_method'],
            discretize_timestamp_method=params['discretize_timestamp_method'],
            discretize_timestamp_parameter=params['discretize_timestamp_parameter'],
            timestamp_mapping=params['timestamp_mapping']
        )

        # Call the preprocessing_encoder method to get the encoded data
        encoded_data = enc.preprocessing_encoder(list_of_pkt, list_of_timestamps)
        print("encoded_data:", encoded_data)

        if not params['combine_pkt_time']:
            # in case the ts was not combined with the pkt, the timestamp information is lost after the gpt
            # -> filter here
            encoded_data = encoded_data[0]

        # decode data again
        decoded_data = []
        for x in encoded_data:
            try:
                decoded_packet = enc.decode(x)
                decoded_data.append(decoded_packet)
            except KeyError as e:
                print(f"KeyError while decoding: {e}")
                # You can handle the error here, e.g., by appending a placeholder value
                decoded_data.append(None)  # Placeholder for failed decoding

        print("decoded_data:", decoded_data)

        checked = []
        for index in range(len(list_of_pkt)):
            orig_pkt = list_of_pkt[index]
            orig_ts = list_of_timestamps[index]

            if decoded_data[index] is None:
                continue

            deco_pkt, deco_ts = decoded_data[index]

            result = enc.are_equal(orig_pkt, orig_ts, deco_pkt, deco_ts)
            checked.append(result)

        print(checked)
        print("All true?:", sum(checked) == len(checked))

        print(f"{combi_id} Testing completed successfully.")
        print()


def test_save_n_load():
    # Set the necessary parameters for testing
    combine_pkt_time = True
    relativize_timestamp_method = 'relative_to_fixed'
    discretize_timestamp_method = 'bin_uniform'
    discretize_timestamp_parameter = {'n_bins': 3}
    timestamp_mapping = False

    # Define a list of packets and timestamps for testing
    np = 3
    list_of_pkt = [generate_packet() for _ in range(np)]
    start_timestamp = int(time.mktime(time.strptime("2020-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")))
    list_of_timestamps = [start_timestamp + i * 1000 for i in range(np)]

    # add identical package
    list_of_pkt.append(list_of_pkt[0])
    list_of_timestamps.append(list_of_timestamps[0])

    # add same package but 1000 later
    list_of_pkt.append(list_of_pkt[0])
    list_of_timestamps.append(list_of_timestamps[0] + 1000)

    # add other package at same time
    list_of_pkt.append(list_of_pkt[1])
    list_of_timestamps.append(list_of_timestamps[0] + 1000)

    # Create an instance of the class to test
    obj = Encoder(combine_pkt_time=combine_pkt_time, relativize_timestamp_method=relativize_timestamp_method,
                  discretize_timestamp_method=discretize_timestamp_method,
                  discretize_timestamp_parameter=discretize_timestamp_parameter, timestamp_mapping=timestamp_mapping)

    # Call the preprocessing_encoder method to get the encoded data
    encoded_data = obj.preprocessing_encoder(list_of_pkt, list_of_timestamps)
    print(encoded_data)
    print([obj.discretize_timestamp(x) for x in list_of_timestamps])
    obj.save_train_data('data/test', encoded_data)

    # Load the variables from the pickle file
    loaded_instance = Encoder.fromfile('data/test/meta.pkl')

    # test if load succ
    print(loaded_instance.feature_mapping_state)

    # test if discretion works
    tmp = [loaded_instance.discretize_timestamp(x) for x in list_of_timestamps]
    print(tmp)

class PrintCounter:
    def __init__(self):
        self.print_count = 0

    def _capture_output(self):
        self.captured_output = []
        with contextlib.redirect_stdout(self):
            test_number_new_pkts()
        return "".join(self.captured_output)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        captured_output = self._capture_output()
        self.print_count = captured_output.count("new item found")

    def write(self, text):
        self.captured_output.append(text)

def test_number_new_pkts(encoder_path, replay_data):
    enc = Encoder.fromfile(encoder_path)
    generator_obj = read_pcap_gz(replay_data)

    #with PrintCounter() as counter:
    for current_index, (current_pkt, current_timestamp) in enumerate(generator_obj):
        enc.encode(current_pkt, current_timestamp)
'''            if current_index == 10:
                print(counter.print_count)
                exit()
    print("lets rutern")
    return counter.print_count'''

replay_data = 'data/attack/attack_4.pcap.gz'
encoder_path = 'out/bin_kmeans/meta.pkl'
print('replay_data', replay_data)
print('encoder_path', encoder_path)

print_count = test_number_new_pkts(encoder_path, replay_data)




# count how many new items are generated

# Run the test
#test_preprocessing_encoder()
#test_single()
#test_save_n_load()
