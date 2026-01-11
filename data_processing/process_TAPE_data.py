import json
import pandas as pd


def process_tape_data(input_file, output_file, label_col):
    with open(input_file, 'r') as fin:
        data = json.load(fin)
        df = pd.DataFrame(data).rename(columns={'primary': 'prot', label_col: 'label'})[['prot', 'label']]
        df['label'] = df['label'].apply(lambda x: x[0])
        df.to_csv(output_file, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Processing of TAPE datasets')
    parser.add_argument('--input_file', help='path to a the json with relevant data')
    parser.add_argument('--output_file', help='Path to output csv file')
    parser.add_argument('--label_col', help='Name of the label column in the input json')
    args = parser.parse_args()
    process_tape_data(args.input_file, args.output_file, args.label_col)