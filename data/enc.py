"""

"""

import os
import pickle
from datetime import datetime, timezone

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

epsilon = 0.02  # epsilon surrounding in which a timestamp is still being accepted as equal (in seconds)
embedding_scaling = 3  # parameter to scale the embedding of the GPT. the higher the value, the more new items are
# allowed until the model crashes

def map_item_to_integer_encoding(item, map_dict, allow_new_items=True):
    """
    maps item du dict entry, if not available it sets it for next and if sets, updates to recent value
    :param allow_new_items: boolean discribes if new entries are allowed, otherwise 0 is the returned encoding
    :param item: item to check
    :param map_dict: dict where information is stored
    :return:
    """
    # currently new items have to be allowed since trainingsdata is not sufficient and new items are to recently
    allow_new_items = True
    if item in map_dict:
        encoding = map_dict[item]
    else:  # token not seen yet
        if allow_new_items:
            encoding = len(map_dict) + 1
            print(f"new item found! {item}")
            map_dict[item] = encoding
        else:
            encoding = 0
    return encoding


class Encoder:
    combine_pkt_time = False
    relativize_timestamp_method = None  # see relativize_timestamp method for more explanation
    relativize_timestamp_state = None  # saves the state for the timestamp processing

    discretize_timestamp_method = None
    discretize_timestamp_parameter = None
    discretize_timestamp_model = None
    decode_discretize_timestamp_state = None

    feature_mapping_state = {}
    timestamp_mapping = False
    timestamp_mapping_state = {}

    pkt_vocab_size = 0
    timestamp_vocab_size = 0

    def __init__(self, combine_pkt_time=True, relativize_timestamp_method=None, discretize_timestamp_method=None,
                 discretize_timestamp_parameter=None, timestamp_mapping=False):
        """
        training constructor, where the different encoding methods are specified. If model is trained use the
        fromfile() method.
        :param combine_pkt_time: boolean, if the packet and timestamp information should be
        combined before encoding
        :param relativize_timestamp_method: one of ['relative_to_fixed',
        'relative_to_syntactic_predecessor', 'relative_to_semantic_predecessor'] where relative_to_fixed means that
        timestamp will be further processed as timestamp, relative_to_syntactic_predecessor means that timestamp is
        distance in microseconds between this timestamp and predecessor, relative_to_semantic_predecessor means that
        time difference between this occurrence and last occurrence of this pkt is taken.
        :param discretize_timestamp_method: one of ['time_feature', 'bin_uniform', 'bin_quantile', 'bin_kmeans'].
        time_feature means that the timestamp is divided into the features specified in discretize_timestamp_parameter
        (requires relativize_timestamp_method='relative_to_fixed'). bin_uniform, bin_quantile, bin_kmeans all
        discretize by dividing data into bins, where uniform: All bins in each feature have identical widths.
        quantile: All bins in each feature have the same number of points. kmeans: Values in each bin have the same
        nearest center of a 1D k-means cluster.
        :param discretize_timestamp_parameter: the parameters required for
        the discretize_timestamp_method. for 'time_feature' this can be alist of all attributes a Datetime object
        has: for example: ['hour', 'minute', 'second', 'microsecond']. For the binning methods the number of clusters
        has to be specified: {'n_bins': 3}.
        """
        self.combine_pkt_time = combine_pkt_time
        self.relativize_timestamp_method = relativize_timestamp_method
        self.discretize_timestamp_method = discretize_timestamp_method
        self.discretize_timestamp_parameter = discretize_timestamp_parameter
        self.timestamp_mapping = timestamp_mapping

    @classmethod
    def fromfile(cls, filepath):
        """constructor to load encriptor from pickel meta file (meta.pkl)"""
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        o = Encoder.__new__(Encoder)
        o.__dict__.update(data)
        return o

    def relativize_timestamp(self, pkt, timestamp):
        """
        depending on the self.relativize_timestamp_method it relatives the timestamp. if it is 'relative_to_fixed' then
        the timestamp does not get processed. if it is 'relative_to_syntactic_predecessor' then return is the time
        difference between the predecessor timestamp which was processed here. if it is
        'relative_to_semantic_predecessor' then the inter-arrival times between the same pkt is returned
        :param pkt: the pkts to which the timestamps belong
        :param timestamp: timestamps to process
        :yield processed timestamp
        """

        # return is the time difference between the predecessor timestamp which was processed here
        if self.relativize_timestamp_method == 'relative_to_syntactic_predecessor':
            if self.relativize_timestamp_state is None:  # init state
                self.relativize_timestamp_state = timestamp
                res = 0.
            else:
                res = timestamp - self.relativize_timestamp_state
                self.relativize_timestamp_state = timestamp

        # inter-arrival times between the same pkt is returned
        elif self.relativize_timestamp_method == 'relative_to_semantic_predecessor':
            if self.relativize_timestamp_state is None:  # init state
                self.relativize_timestamp_state = {pkt: timestamp}
                res = 0.
            elif pkt not in self.relativize_timestamp_state.keys():  # pkt first time seen
                self.relativize_timestamp_state[pkt] = timestamp
                res = 0.
            else:  # pkt was observed earlier
                res = timestamp - self.relativize_timestamp_state[pkt]
                self.relativize_timestamp_state[pkt] = timestamp

        # returns the plain timestamp
        elif self.relativize_timestamp_method == 'relative_to_fixed':
            res = timestamp

        return res

    def train_discretize_timestamp_model(self, list_of_timestamps):
        if 'bin' in self.discretize_timestamp_method:
            method_to_strategy = {
                'bin_uniform': 'uniform',
                'bin_quantile': 'quantile',
                'bin_kmeans': 'kmeans'
            }
            discretizer = KBinsDiscretizer(n_bins=self.discretize_timestamp_parameter['n_bins'], encode='ordinal',
                                           strategy=method_to_strategy[self.discretize_timestamp_method],
                                           subsample=None)
            timestamps = [[timestamp] for timestamp in list_of_timestamps]
            discretizer.fit(timestamps)

            bin_intervals = discretizer.bin_edges_
            # store intervals of bins to decode later
            self.decode_discretize_timestamp_state = bin_intervals[0].tolist()
            # print(self.decode_discretize_timestamp_state)

            # Save the trained model
            self.discretize_timestamp_model = discretizer
        elif self.discretize_timestamp_method == 'time_feature':
            # TODO
            return
        else:
            raise NotImplementedError(f"{self.discretize_timestamp_method}")

    def discretize_timestamp(self, timestamp):
        """
        :param timestamp: timestamp to discretize
        :return:
        """
        if 'bin' in self.discretize_timestamp_method:
            discretized_timestamp = self.discretize_timestamp_model.transform([[timestamp]])
            return int(discretized_timestamp.flatten().tolist()[0])

        elif self.discretize_timestamp_method == 'time_feature':
            dt = datetime.fromtimestamp(timestamp)
            res = []
            for param in self.discretize_timestamp_parameter:
                try:
                    res.append(getattr(dt, param))
                except AttributeError:
                    print(f"log: cant find attribute {param} in Datetime-object, skipping")
            return tuple(res)

        else:
            raise NotImplementedError(f"{self.discretize_timestamp_method}")

    def preprocessing_encoder(self, list_of_pkt, list_of_timestamps=None):
        """
        function to process and encode the training data. Also trains required models to do so.
        :param list_of_pkt: list of pkt to encode for training
        :param list_of_timestamps: list of timestamps respectively
        :return:
        """
        # print(f"INPUT: list of timestamps: {list_of_timestamps}")
        if self.relativize_timestamp_method is not None:
            list_of_timestamps = [self.relativize_timestamp(pkt, timestamp) for pkt, timestamp in
                                  zip(list_of_pkt, list_of_timestamps)]
            # print(f"After relativization: {list_of_timestamps}")

        if self.discretize_timestamp_method is not None:
            self.train_discretize_timestamp_model(list_of_timestamps)
            list_of_timestamps = [self.discretize_timestamp(x) for x in list_of_timestamps]
            # print(f"After discretion: {list_of_timestamps}")

        # print(f"INPUT: list of pkt: {list_of_pkt}")
        if self.combine_pkt_time:
            list_of_pkt = [(*pkt, timestamp) for pkt, timestamp in zip(list_of_pkt, list_of_timestamps)]
            # print(f"After combination: {list_of_pkt}")

        # map features to integer
        encoding = [map_item_to_integer_encoding(x, self.feature_mapping_state, allow_new_items=True) for x in list_of_pkt]
        # print(f"After encoding: {encoding}")

        if self.timestamp_mapping:  # encode timestamp as well but separately

            timestamp_encoding = [
                map_item_to_integer_encoding(x, self.timestamp_mapping_state, allow_new_items=True) for x in list_of_timestamps]
        else:  # timestamp is already encoded
            timestamp_encoding = list_of_timestamps
        # print(f"After timestamp encoding: {timestamp_encoding}")

        # return
        if self.combine_pkt_time or self.relativize_timestamp_method is None:
            return encoding
        else:
            return encoding, timestamp_encoding

    def save_train_data(self, data_directory_path, list_of_pkt, list_of_timestamps=None, split=0.9):
        """
        function to save the obtained models for preprocessing and the data for training of the GPT model
        :param data_directory_path: directory where to safe the data
        :param list_of_pkt: list of encoded packets to save
        :param list_of_timestamps: list of timestamps to save
        :param split: fraction which will be declared as train data
        """

        # Create the directory if it doesn't exist
        os.makedirs(data_directory_path, exist_ok=True)

        # split into train and test
        n = len(list_of_pkt)
        train_list_of_pkt = list_of_pkt[:int(n * split)]
        val_list_of_pkt = list_of_pkt[int(n * split):]

        # export to bin files
        train_list_of_pkt = np.array(train_list_of_pkt, dtype=np.uint16)
        val_list_of_pkt = np.array(val_list_of_pkt, dtype=np.uint16)
        train_list_of_pkt.tofile(os.path.join(data_directory_path, 'pkt_train.bin'))
        val_list_of_pkt.tofile(os.path.join(data_directory_path, 'pkt_val.bin'))

        if list_of_timestamps is not None:  # again for timestamps
            # split into train and test
            train_list_of_timestamps = list_of_timestamps[:int(n * split)]
            val_list_of_timestamps = list_of_timestamps[int(n * split):]

            # export to bin files
            train_list_of_timestamps = np.array(train_list_of_timestamps, dtype=np.uint16)
            val_list_of_timestamps = np.array(val_list_of_timestamps, dtype=np.uint16)
            train_list_of_timestamps.tofile(os.path.join(data_directory_path, 'timestamp_train.bin'))
            val_list_of_timestamps.tofile(os.path.join(data_directory_path, 'timestamp_val.bin'))

        #  vocabulary embedding
        self.pkt_vocab_size = (len(self.feature_mapping_state)+1)*embedding_scaling
        self.timestamp_vocab_size = (len(self.timestamp_mapping_state)+1)*embedding_scaling

        # save the meta information as well, to help us encode/decode later
        with open(os.path.join(data_directory_path, 'meta.pkl'), 'wb') as f:
            pickle.dump(self.__dict__, f)

    def encode(self, pkt, timestamp=None):
        """
        encode function for runtime usage, does not have training steps, operates at a per package level
        :param pkt:
        :param timestamp:
        :return:
        """
        # print(f"INPUT: list of timestamps: {list_of_timestamps}")
        if self.relativize_timestamp_method is not None:
            timestamp = self.relativize_timestamp(pkt, timestamp)
            # print(f"After relativization: {list_of_timestamps}")

        if self.discretize_timestamp_method is not None:
            timestamp = self.discretize_timestamp(timestamp)
            # print(f"After discretion: {list_of_timestamps}")

        # print(f"INPUT: list of pkt: {list_of_pkt}")
        if self.combine_pkt_time:
            pkt = (*pkt, timestamp)
            # print(f"After combination: {list_of_pkt}")

        # map features to integer
        encoding = map_item_to_integer_encoding(pkt, self.feature_mapping_state, allow_new_items=False)
        # print(f"After encoding: {encoding}")

        if self.timestamp_mapping:  # encode timestamp as well but separately
            timestamp_encoding = map_item_to_integer_encoding(timestamp, self.timestamp_mapping_state, allow_new_items=False)
        else:  # timestamp is already encoded
            timestamp_encoding = timestamp
        # print(f"After timestamp encoding: {timestamp_encoding}")

        # return
        if self.combine_pkt_time or self.relativize_timestamp_method is None:
            return encoding
        else:
            return encoding, timestamp_encoding

    def decode(self, encoding):
        # print(encoding)

        pkt = encoding

        # demap pkt
        try:
            pkt = [k for k, v in self.feature_mapping_state.items() if v == pkt][0]
            # print(pkt)
        except:
            # print(f"can't decode {pkt}, no such key in mapping")
            raise KeyError(f"can't decode {pkt}, no such key in mapping")
        # print(pkt)

        if self.combine_pkt_time:
            pkt, timestamp = pkt[:-1], pkt[-1]
        else:
            # when pkt info was not combined, timestamp was used as position arguement,
            # so no timestamp information left after gpt
            return pkt, None

        # de timestamp mapping
        if self.timestamp_mapping:
            try:
                timestamp = [k for k, v in self.timestamp_mapping_state.items() if v == timestamp][0]
            except:
                # print(f"can't decode timestamp: {timestamp}, no such key in timestamp mapping")
                raise KeyError(f"can't decode timestamp: {timestamp}, no such key in timestamp mapping")

        # print(pkt, timestamp)

        # ['time_feature', 'bin_uniform', 'bin_quantile', 'bin_kmeans']
        if self.discretize_timestamp_method is None:
            pass
        elif self.discretize_timestamp_method == 'time_feature':
            attributes = {x: y for x, y in zip(self.discretize_timestamp_parameter, timestamp)}

            # Default values for missing attributes
            default_attributes = {
                'year': 1970,
                'month': 1,
                'day': 1,
                'tzinfo': timezone.utc
            }
            attributes.update(default_attributes)
            dt = datetime(**attributes)
            timestamp = dt.timestamp()

        elif 'bin' in self.discretize_timestamp_method:
            bin_number = timestamp
            bin_edges = self.decode_discretize_timestamp_state

            if 0 <= bin_number < len(bin_edges) - 1:
                timestamp = (bin_edges[bin_number], bin_edges[bin_number + 1])
            # elif bin_number == len(bin_edges) - 1:
            # timestamp = f"({bin_edges[bin_number]}, infinity)"
            else:
                timestamp = "bin out of range"
        else:
            raise NotImplementedError(self.discretize_timestamp_method)

        # TODO relativization
        # ['relative_to_fixed','relative_to_syntactic_predecessor', 'relative_to_semantic_predecessor']
        # print(pkt, timestamp)
        return pkt, timestamp

    def are_equal(self, orig_pkt, orig_ts, deco_pkt, deco_ts):
        """
        function to compare an original pkt with timestamp to a decoded one.
        i.e., it checks if the timestamp fits the intervals of the decoded version
        :param orig_pkt: original pkt as tupel: ('src', 'dst', 'fcode')
        :param orig_ts: timestamp of original pkt
        :param deco_pkt: the decoded version of the pkt as tupel: ('src', 'dst', 'fcode')
        :param deco_ts: the decoded version of the timestamp as returned by the @decode function
        :return: a number between 0 and 1, where 0 means pkt are different, 1 means pkts are equal and
        """
        if not orig_pkt == deco_pkt:
            return 0
        if deco_ts is None:
            return 1

        # assume that ts is not relativized:
        # can lead to mistakes when the state has to be updated
        orig_ts = self.relativize_timestamp(orig_pkt, orig_ts)

        if self.discretize_timestamp_method == 'time_feature':
            # create epsilon environment where timestamp is valid
            interval_low, interval_high = deco_ts - epsilon, deco_ts + epsilon
            ts_check = orig_ts >= interval_low and orig_ts >= interval_high
            return ts_check
        elif 'bin' in self.discretize_timestamp_method:
            # check if ts is in interval
            interval_low, interval_high = deco_ts[0], deco_ts[1]
            ts_check = interval_low <= orig_ts <= interval_high
            return ts_check
        elif not self.discretize_timestamp_method is None:
            raise NotImplementedError(self.discretize_timestamp_method)
        else:  # no dis used
            raise NotImplementedError("the equal method is not yet implemented for this encoder combination")
        print("not possible?")

    def get_state(self):
        configuration = {
            'combine_pkt_time': self.combine_pkt_time,
            'relativize_timestamp_method': self.relativize_timestamp_method,
            'relativize_timestamp_state': self.relativize_timestamp_state,
            'discretize_timestamp_method': self.discretize_timestamp_method,
            'discretize_timestamp_parameter': self.discretize_timestamp_parameter,
            'discretize_timestamp_model': self.discretize_timestamp_model,
            'feature_mapping_state': self.feature_mapping_state,
            'timestamp_mapping': self.timestamp_mapping,
            'timestamp_mapping_state': self.timestamp_mapping_state,
            'pkt_vocab_size': self.pkt_vocab_size,
            'timestamp_vocab_size': self.timestamp_vocab_size
        }
        return configuration
