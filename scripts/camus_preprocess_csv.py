import pandas as pd
import os
import argparse

"""
This script extracts and consolidates the numerical data (AP2 ED frame, AP2 ES frame, AP2 total number of frames, AP4 ED
frame, AP4 ES frame, AP4 total number of frames, and EF) from the CAMUS dataset folder (training or testing) into one
.csv file, 'CAMUS_extracted_train.csv'.

To run: python camus_preprocess_csv.py --dataset_path <path_to_CAMUS_dataset_folder>
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True,
                        help="Path to the folder containing the CAMUS dataset")
    args = parser.parse_args()

    # input path to CAMUS training data
    dataset_path = args.dataset_path

    # create table for data
    dataset_table = []

    # iterate through list of patients
    patient_list = os.listdir(dataset_path)

    for patient in patient_list:
        patient_path = dataset_path + '/' + patient
        patient_files = os.listdir(patient_path)

        # extract AP2 information
        patient_path_AP2 = patient_path + '/Info_2CH.cfg'
        if 'Info_2CH.cfg' in patient_files:
            AP2 = True
            f = open(patient_path_AP2)
            AP2_info = f.readlines()
            AP2_ED = int(AP2_info[0].split(' ')[1])  # extract ED frame into int
            AP2_ES = int(AP2_info[1].split(' ')[1])  # extract ES frame into int
            AP2_totalFrames = int(AP2_info[2].split(' ')[1])  # extract number of frames into int
            AP2_EF = float(AP2_info[8].split(' ')[1])  # extract EF into float
        else:
            #print(patient + ': No Info_2CH')
            AP2 = False

        # extract AP4 information
        patient_path_AP4 = patient_path + '/Info_4CH.cfg'
        if 'Info_4CH.cfg' in patient_files:
            AP4 = True
            f = open(patient_path_AP4)
            AP4_info = f.readlines()
            AP4_ED = int(AP4_info[0].split(' ')[1])  # extract ED frame into int
            AP4_ES = int(AP4_info[1].split(' ')[1])  # extract ES frame into int
            AP4_totalFrames = int(AP4_info[2].split(' ')[1])  # extract number of frames into int
            AP4_EF = float(AP4_info[8].split(' ')[1])  # extract EF into float
        else:
            #print(patient + ': No Info_4CH')
            AP4 = False

        if AP2 and AP4:
            # confirm EF is the same in both AP2 and AP4, otherwise print error
            assert AP2_EF == AP4_EF

            # enter patient row information
            patient_row = [patient, AP2_ED, AP2_ES, AP2_totalFrames,
                           AP4_ED, AP4_ES, AP4_totalFrames, AP4_EF]
            dataset_table.append(patient_row)

    # create and populate dataframe
    data_pd = pd.DataFrame(dataset_table)
    data_pd.columns = ['PatientID', 'AP2_ED_frame', 'AP2_ES_frame', 'AP2_total_frames',
                       'AP4_ED_frame', 'AP4_ES_frame', 'AP4_total_frames', 'EF']

    # Save as .csv
    data_pd.to_csv('CAMUS_extracted_train.csv', index=False)
