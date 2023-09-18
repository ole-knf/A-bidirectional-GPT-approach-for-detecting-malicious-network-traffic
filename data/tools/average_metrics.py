import json

# Define a list of file paths
dir = 'data/evaluation/'

approach = 'bin_kmeans'

# Initialize variables to store the sum of values

total_accuracy = 0
total_precision = 0
total_recall = 0
total_F1 = 0
total_F01 = 0
total_detected = 0
total_eTaF01 = 0
total_eTaF1 = 0
total_eTaR = 0
total_eTaP = 0

file_amount = 4
# Loop through the files and sum the values
for number in range(1,file_amount+1):
    with open(f"{dir}attack_{number}/{approach}.txt", 'r') as file:
        data = json.load(file)
        total_accuracy += data["Accuracy"]
        total_precision += data["Precision"]
        total_recall = data["Recall"]
        total_detected += data["Detected-Scenarios-Percent"]
        total_F1 += data["F1"]
        total_F01 += data["F0.1"]
        total_eTaF01 += data["eTaF0.1"]
        total_eTaF1 += data["eTaF1"]
        total_eTaR += data["eTaR"]
        total_eTaP += data["eTaP"]

# Calculate the average values
total_accuracy = total_accuracy / file_amount
total_precision = total_precision / file_amount
total_recall = total_recall / file_amount
total_F1 = total_F1 / file_amount
total_F01 = total_F01 / file_amount
total_detected = total_detected / file_amount
total_eTaF01 = total_eTaF01 / file_amount
total_eTaF1 = total_eTaF1 / file_amount
total_eTaR = total_eTaR / file_amount
total_eTaP = total_eTaP / file_amount

# Create a new JSON object with the averaged values
averaged_data = {
    "total_accuracy": total_accuracy,
    "total_precision": total_precision,
    "total_recall": total_recall,
    "total_F1": total_F1,
    "total_F01": total_F01,
    "total_detected": total_detected,
    "total_eTaF01": total_eTaF01,
    "total_eTaF1": total_eTaF1,
    "total_eTaR": total_eTaR,
    "total_eTaP": total_eTaP
}

# Print the averaged data or save it to a file
print(json.dumps(averaged_data, indent=4))
