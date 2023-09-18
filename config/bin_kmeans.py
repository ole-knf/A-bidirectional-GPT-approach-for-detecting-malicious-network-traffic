project = 'bin_kmeans'

""" 
DATA
"""
# data where the model is trained on
train_data = 'data/train/normal.pcap.gz'
# processed data, models and output will be saved in {out_dir}/{project}
out_dir = 'out'

"""
System
"""
# ['cpu', 'cuda']
device = 'cuda'
compile = True  # do not torch compile the model
# TODO can be empty?
seed = 1337

"""
Steps
"""
force_preprocessing = False
force_train_forwards = False
force_train_backwards = False

"""
Preprocessing
"""
# Set the necessary parameters for testing
# True, False
combine_pkt_time = True
# ['relative_to_fixed', 'relative_to_syntactic_predecessor', 'relative_to_semantic_predecessor']
relativize_timestamp_method = 'relative_to_syntactic_predecessor'
# ['time_feature', 'bin_uniform', 'bin_quantile', 'bin_kmeans']
discretize_timestamp_method = 'bin_kmeans'
# ['hour', 'minute', 'second', 'microsecond'] or {'n_bins': 20}
discretize_timestamp_parameter = {'n_bins': 20}
timestamp_mapping = False

"""
train / GPT config
"""
wandb_log = False
wandb_project = 'train_testing'
wandb_run_name = 'mini-gpt'
always_save_checkpoint = False
gradient_accumulation_steps = 1
eval_interval = 200  # keep frequent because we'll overfit
eval_iters = 100
log_interval = 25  # don't print too too often
batch_size = 64
block_size = 5
# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 128
dropout = 0.1
learning_rate = 1e-4  # with baby networks can afford to go a bit higher
max_iters = 4000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = learning_rate/10  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
warmup_iters = 100  # not super necessary potentially

