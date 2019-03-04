import csv


def import_data():
    with open('unknown_hmm_multi_logs.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            print(row)
