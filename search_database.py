import numpy as np
from argparse import ArgumentParser
import json


def json_loader(path):
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data


def search_database(modulus, strength, deformation, database_path='database/database.json'):
    y_tar = np.array([modulus, strength, deformation])
    database = json_loader(database_path)
    dist = np.sum((y_tar[None, :] - database['tissue'])**2, axis=-1)
    formula = database['formula'][np.argmin(dist)]
    return formula


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--m', type=float, help='Elastic modulus')
    parser.add_argument('--s', type=float, help='Tensile strength')
    parser.add_argument('--d', type=float, help='Deformation')
    parser.add_argument('--database_path', type=str, default='database/database.json')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    formula_keys = ['SP1', 'SP2', 'HP1', 'HP2', 'SC', 'HC']
    args = parse_args()
    formula = search_database(args.m, args.s, args.d, args.database_path)
    for key, f in zip(formula_keys, formula):
        print(f'{key}: {f * 100:.2f}')