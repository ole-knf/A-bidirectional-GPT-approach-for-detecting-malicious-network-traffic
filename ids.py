'''
Evaluates a file based on a chose backwards and forwards model, packet by packet starting at packet 0.
If option 'verbose' is set, every packet with a score lower than the defined alarm threshold will be printed to the console.
'''
from datetime import datetime, timedelta
import os
from contextlib import nullcontext
from collections import deque
import torch
from data.enc import Encoder
from model import GPTConfig, GPT
from data.pcap2enc import read_pcap_gz
import json

def torch_init():
    '''
    Sets up all torch components neccessary for predictions
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    return ctx


def fetch_model(checkpoint_name):
    """
    Retrieves pretrained models and encoders for a given direction
    """
    ckpt_path = os.path.join(out_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    return model


def predict_packet(org_pkt, org_timestamp, sequence, encoder, eval_model, direction):
    """
    Predicts multiple packets with one of the models and returns the percentage of the actual packet following
    """

    # return if state is not big enough
    if len(sequence) < number_pkts_to_read:
        return -1

    x = (torch.tensor(sequence, dtype=torch.long, device=device)[None, ...])

    with torch.no_grad():
        with ctx:
            y, idx_next, probs = eval_model.generate(x, max_new_tokens, temperature=temperature)
            predicted_seq_decoded = []
            for pkt in y[0].tolist():
                try:
                    predicted_seq_decoded.append(encoder.decode(pkt))
                except KeyError:
                    predicted_seq_decoded.append(None)
            probs = probs[0].tolist()

    predicted_seq_decoded = predicted_seq_decoded[number_pkts_to_read:]
    score = 0
    for prediction in predicted_seq_decoded:
        if prediction is None:
            continue
        else:
            predict_pkt, predict_timestamp = prediction[0], prediction[1]
        check = encoder.are_equal(org_pkt, org_timestamp, predict_pkt, predict_timestamp)
        if check:
            score += probs[predicted_seq_decoded.index(prediction)]
            break

    return score


def evaluate_scores(f_score, b_score):
    """
    function which calculates the score for packets, manages the number_pkts_to_read offset aswell
    :return: pkt, timestamp, relative score
    """
    # If packet could not be predicted, double the score from other model
    if f_score == -1 and b_score == -1:
        return -1
    if f_score < 0:
        b_score *= 2
        f_score = 0
    if b_score < 0:
        f_score *= 2
        b_score = 0

    score_percentage = (f_score + b_score) / 2

    return score_percentage

def update_state(current_pkt, current_timestamp, current_encodeded, score_forward):
    f_seq_state.append(current_encodeded)
    b_seq_state.appendleft(current_encodeded)
    tmp = prd_state.popleft()
    eval_pkt, eval_timestamp, eval_f_score = tmp[0]
    prd_state.append(((current_pkt, current_timestamp, score_forward),))
    return eval_pkt, eval_timestamp, eval_f_score


def load_attackfile():
    """
    Loads the data of an attack file which contains start and end time for each attack in epoch time
    """

    try:
        with open(attack_meta_data, "r") as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"An error occurred while loading the JSON file: {e}")
        return None


def calculate_metrics(times, attack_intervals, malicious):
    """
    Calculation of basic metrics of evaluation, True Positive/Negative, False Positive/Negative
    These can be assembled into more compelx metrics from ipal eval, like accuracy or precision
    """
    metrics = {"TN": 0, "TP": 0, "FN": 0, "FP": 0}
    for i in range(len(times)):
        in_interval = False
        for interval in attack_intervals:
            start_time = interval["start"]
            end_time = interval["end"]
            if times[i] >= start_time and times[i] <= end_time:
                in_interval = True

        if in_interval:
            if malicious[i]:
                metrics["TP"] += 1
            else:
                metrics["FN"] += 1
        else:
            if malicious[i]:
                metrics["FP"] += 1
            else:
                metrics["TN"] += 1

    for metric in metrics:
        metrics[metric] = metrics[metric] / len(times)

    return metrics

def ipal_format():
    """
    Transforms the evaluated output into an readble format for ipal evaluate
    """
    attack_intervals = load_attackfile()
    ipal_enconding = []
    for packet in packets_evaluated:   
        in_interval = False
        for interval in attack_intervals:
            start_time = interval["start"]
            end_time = interval["end"]
            if packet[0] >= start_time and packet[0] <= end_time:
                in_interval = True
                break

        ipal_enconding.append((packet[0], str(in_interval).lower(), str(packet[2]).lower()))
    ipal_file_name = "GPT_"+os.path.splitext(os.path.basename(attack_meta_data))[0]+".ipal"

    with open(os.path.join(out_dir,ipal_file_name), "w") as file:
        for i, entry in enumerate(ipal_enconding):
            line = f'"id": {i}, "timestamp": {entry[0]}, "malicious": {entry[1]}, "ids": {entry[2]}'
            file.write('{' + line + '}\n')



def generate_overview():
    """
    Takes the evaluated data and the information of the specified attack file to generate a graph.
    It contains a visualization of the packet scores over time. Every packet flagged as malicious is marked as a red dot.
    The real attack intervals contained in the attack file are representaed als grey areas over the graph
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import datetime

    attack_intervals = load_attackfile()
    times = [item[0] for item in packets_evaluated]
    scores = [item[1] * 100 for item in packets_evaluated]
    malicious = [item[2] for item in packets_evaluated]

    metrics = calculate_metrics(times, attack_intervals, malicious)

    times = [(datetime.datetime.fromtimestamp(item)).strftime('%H:%M:%S.%f') for item in
             times]  # representation in local time zone

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(times, scores, color='blue', label='Score')
    ax1.scatter([t for i, t in enumerate(times) if malicious[i]], [s for i, s in enumerate(scores) if malicious[i]],
                color='red', label='Malicious')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Score in  %', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    for interval in attack_intervals:
        start_time = (datetime.datetime.fromtimestamp(interval["start"])).strftime('%H:%M:%S.%f')
        end_time = (datetime.datetime.fromtimestamp(interval["end"])).strftime('%H:%M:%S.%f')
        ax1.axvspan(start_time, end_time, alpha=0.2, color='gray')

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())  # minticks=4, maxticks=10
    ax1.xaxis.set_minor_locator(mdates.AutoDateLocator())  # minticks=8, maxticks=30

    plt.xticks(rotation=30)
    plt.subplots_adjust(bottom=0.2)
    ax1.legend(loc='upper left')
    legend_pos = ax1.get_legend().get_bbox_to_anchor().get_points()[0]  # set metric underneth the legend
    x_pos = (legend_pos[0] / 1000) - 0.05
    y_pos = 1 - (legend_pos[1] / 1000) - 0.2
    plt.figtext(x_pos, y_pos, "".join([f"{key}:{round(metrics[key] * 100, 3)}%\n" for key in metrics]), fontsize=12,
                color='black', ha='left')
    plt.title('Score and Intervals')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/overview-{os.path.splitext(os.path.basename(attack_meta_data))[0]}.png", dpi=fig.dpi)
    plt.show()


if __name__ == "__main__":
    project = 'dev'
    # processed data, models and output will be saved in {out_dir}/{project}
    out_dir = 'out'
    # general params
    replay_data = 'data/attack/attack_1.pcap.gz'
    attack_meta_data = "data/attack/attack_1.json"
    # The amount of packets in the pcap. Is the calculation basis for the remaining time
    number_of_pkts_to_process = 5500000

    # model params
    number_pkts_to_read = 128
    max_new_tokens = 1
    temperature = 0.8
    seed = 1337
    device = 'cuda'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile = True

    # IDS params
    verbose = True
    logging_interval = 100
    alarm_threshold = 0  # percentage, any score lower will be flagged 
    overview = True
    ipal = True

    n_embd = 64

    # output_path = "attack/"
    # overide params from CLI
    exec(open('configurator.py').read())

    number_pkts_to_read = n_embd

    assert ((overview and attack_meta_data != "") or not overview)

    out_dir = os.path.join(out_dir, project)

    f_seq_state = deque(maxlen=number_pkts_to_read)
    b_seq_state = deque(maxlen=number_pkts_to_read)
    
    prd_state = deque(maxlen=number_pkts_to_read+1)
    for _ in range(number_pkts_to_read+1):
        prd_state.append(((0, 0, -1),))

    # Fetch torch backend Model, Encoder, and Data
    print("Initialize Torch backend...")
    ctx = torch_init()
    print("Fetch models...")
    forward_model = fetch_model("forward.pt")
    backward_model = fetch_model("backward.pt")
    meta_path = os.path.join(out_dir, 'meta.pkl')
    assert (os.path.exists(meta_path))
    enc = Encoder.fromfile(meta_path)
    print(f"loaded encoder with: {enc.get_state()}")
    print("Fetch data...")
    # generator which yields (ip.src, ip.dst, mb.funcCode), float(packet.time)
    generator_obj = read_pcap_gz(replay_data)

    # Iterate over all packets and predict the value
    print("Start evaluation...")

    alarm_count = 0
    start_time = datetime.now().timestamp()
    avg_t_f_p = 0

    if overview or ipal:
        packets_evaluated = []

    eval_pkt, eval_timestamp, eval_f_score = 0, 0, -1
    for current_index, (current_pkt, current_timestamp) in enumerate(generator_obj):
        time_for_processing = datetime.now().timestamp() - start_time
        avg_t_f_p += time_for_processing
        start_time = datetime.now().timestamp()

        if current_index % logging_interval == 0 and current_index != 0:
            print(f"INFO: pkt {current_index}, lst_prc_time: {round(time_for_processing, 4)}, avg: {round(avg_t_f_p/(current_index-1), 4)}, finish: {(datetime(1,1,1)+timedelta(seconds=(number_of_pkts_to_process-current_index)*avg_t_f_p/(current_index-1))).strftime('%d Days, %H:%M:%S')}")
            # The calculation for the remaining time is off by around 1 1/2 days. We don't really now why.
        current_encodeded = enc.encode(current_pkt, current_timestamp)
        
        try:
            score_forward = predict_packet(current_pkt, current_timestamp, f_seq_state, enc, forward_model, "forward.pt")
            score_backward = predict_packet(eval_pkt, eval_timestamp, b_seq_state, enc, backward_model, "backward.pt")
        except Exception as e:
            print("current", current_index, current_pkt, current_timestamp, current_encodeded)
            print("state", f_seq_state, b_seq_state, prd_state)
            print("eval", eval_pkt, eval_timestamp, eval_f_score)
            print(enc.get_state())
            print("An exception occurred:", e)
            exit()

        eval_score = evaluate_scores(eval_f_score, score_backward)

        if current_index > number_pkts_to_read+1:
            alarm = eval_score <= alarm_threshold

            # Print Alarm
            if verbose and alarm:
                print(
                    f"ALARM {alarm_count} (rate:{alarm_count/current_index}): The following packet ({current_index}) seems malicious:\n Packet{eval_pkt} at {eval_timestamp} with a score of {eval_score}%", eval_f_score, score_backward)
                alarm_count += 1
            else:
                pass
            if overview or ipal:
                packets_evaluated.append((eval_timestamp, eval_score, alarm))
        else:
            print(f"Number of pkt to read is not full yet {number_pkts_to_read-current_index} left. skipping {current_pkt, current_timestamp}")

        # Update the states for next pkt
        eval_pkt, eval_timestamp, eval_f_score = update_state(current_pkt, current_timestamp, current_encodeded, score_forward)

    if ipal:
        print("Creating ipal file...")
        ipal_format()
    
    if overview:
        generate_overview()