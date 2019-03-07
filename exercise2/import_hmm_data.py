import csv
import numpy as np


def import_data():
    data = {
        "unknown_hmm_multi_logs": [],
        "nominal_hmm_multi_logs": [],
        "ThreeClass_log": []
    }
    with open('unknown_hmm_multi_logs.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data["unknown_hmm_multi_logs"].append([int(v) - 1 for v in row])
    with open('nominal_hmm_multi_logs.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data["nominal_hmm_multi_logs"].append([int(v) - 1 for v in row])
    with open('ThreeClass_log.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data["ThreeClass_log"].append([float(v) for v in row])
    for key, value in data.items():
        data[key] = np.array(value)
    return data
