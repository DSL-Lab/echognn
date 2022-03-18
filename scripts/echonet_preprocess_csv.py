import pandas as pd
import argparse

"""
This script uses the frame locations in VolumeTracings to add ES and ED frame location columns to FileList.csv and save
the new CSV.
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv_path', required=True,
                        help="Path to CSV file containing EF labels for EchoNet")
    parser.add_argument('--tracing_csv_path', required=True,
                        help='Path to CSV file containing Volume Tracings for EchoNet')
    parser.add_argument('--output_dir', type=str, default='./output.csv',
                        help='Output CSV file')
    args = parser.parse_args()

    data_csv_path = args.data_csv_path
    tracing_csv_path = args.tracing_csv_path
    output_dir = args.output_dir

    # Load data csv file
    data_df = pd.read_csv(data_csv_path)

    # Load tracings csv file
    tracing_df = pd.read_csv(tracing_csv_path)

    file_names = data_df['FileName'].tolist()

    # Add columns for ED and ES
    data_df['ESFrame'] = ''
    data_df['EDFrame'] = ''

    for file_name in file_names:
        try:
            es_ed_idx = tracing_df.loc[tracing_df['FileName'] == file_name + '.avi', 'Frame']

            data_df.loc[data_df['FileName'] == file_name, 'EDFrame'] = es_ed_idx.iloc[0]
            data_df.loc[data_df['FileName'] == file_name, 'ESFrame'] = es_ed_idx.iloc[-1]
        except IndexError:
            print("Volume Tracing is missing for {}".format(file_name))

    # Save the new csv
    data_df.to_csv(output_dir)
