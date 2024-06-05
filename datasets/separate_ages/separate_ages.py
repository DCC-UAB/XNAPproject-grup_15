import csv
from collections import defaultdict

AFAD_PATH_train = 'datasets/afad-propi-test.csv'
AFAD_PATH_test = 'datasets/afad-propi-test.csv'
age_ranges_afad = [(0, 6), (7, 13), (14, 20), (21, 27), (28, 30)]
# GROUPS ------------1--------2--------3---------4---------5

CACD_PATH_train = 'datasets/cacd-train1.csv'
CACD_PATH_test = 'datasets/cacd-train1.csv'

age_ranges_cacd = [(0, 7), (8, 15), (16, 23), (24, 31), (32, 39), (40, 46)]
# GROUPS ------------1--------2--------3---------4---------5---------6  

def split_dataset_by_age_groups_AFAD(input_file, prefix,age_ranges):
    grouped_data = defaultdict(list)

    with open(input_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            age = int(row['age'])
            
            for i, (start, end) in enumerate(age_ranges):
                if start <= age <= end:
                    grouped_data[i].append(row)
                    break

    for i, group_data in grouped_data.items():
        filename = f'datasets/separate_ages/{prefix}_grup_{i+1}.csv'
        with open(filename, mode='w', newline='') as file:
            fieldnames = ['file', 'path', 'age', 'gender']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_data)

def split_dataset_by_age_groups_CACD(input_file, prefix,age_ranges):
    grouped_data = defaultdict(list)

    with open(input_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            age = int(row['age'])
            
            for i, (start, end) in enumerate(age_ranges):
                if start <= age <= end:
                    grouped_data[i].append(row)
                    break

    for i, group_data in grouped_data.items():
        filename = f'datasets/separate_ages/{prefix}_grup_{i+1}.csv'
        with open(filename, mode='w', newline='') as file:
            fieldnames = ['file', 'age','']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_data)

# Aplicar la función para los conjuntos de entrenamiento y prueba
split_dataset_by_age_groups_AFAD(AFAD_PATH_train, 'afad_train', age_ranges_afad)
split_dataset_by_age_groups_AFAD(AFAD_PATH_test, 'afad_test', age_ranges_afad)

split_dataset_by_age_groups_CACD(CACD_PATH_train, 'cacd_train', age_ranges_cacd)
split_dataset_by_age_groups_CACD(CACD_PATH_test, 'cacd_test', age_ranges_cacd)

print("Los archivos CSV para cada grupo y conjunto han sido creados.")
