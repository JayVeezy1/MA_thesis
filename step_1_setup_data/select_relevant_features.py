import os
from operator import countOf

import numpy as np
import pandas as pd
from os.path import isfile

from matplotlib import pyplot as plt


def export_final_dataset(project_path, use_case_name):
    # step 0: select relevant features
    # select all known general patient_info features
    general_features: list = ['icustay_id', 'hadm_id', 'subject_id', 'intime', 'outtime', 'los_hours', 'dbsource',
                              'icustays_count', 'age', 'patientweight', 'gender', 'ethnicity', 'admission_type',
                              'discharge_location', 'insurance', 'language', 'religion', 'marital_status',
                              'diagnosis_text', 'dob', 'dod', 'death_in_hosp', 'death_3_days', 'death_30_days',
                              'death_180_days', 'death_365_days', 'preiculos', 'gcs', 'mechvent', 'electivesurgery',
                              'stroke_type', 'infarct_type', 'hypertension_flag', 'diabetes_flag', 'cancer_flag',
                              'obesity_flag', 'drug_abuse_flag', 'sepsis_flag', 'icd9_code', 'all_icd9_codes']

    # select features that are known and important from prev research (will be added again later)
    vitals_features: list = ['charttime',
                             # selected following the novel nomogram paper:
                             # Vitals
                             # 'Anion gap',      # removed: same as 'Anion Gap' but 1 hour later
                             'Anion Gap', 'Bicarbonate', 'Chloride (whole blood)', 'Calcium Total', 'Creatinine',
                             'Glucose (whole blood)', 'Potassium (whole blood)', 'Sodium (whole blood)', 'Hemoglobin',
                             # 'WBC',            # removed: same as 'White Blood Cells' but 1 hour later
                             'White Blood Cells', 'Packed Red Blood Cells', 'Platelet Count', 'Prothrombin time',
                             'INR',  # definition: international normalized ratio (INR) for blood
                             'Heart Rate', 'Respiratory Rate',
                             # 'Respiratory Rate (Total)',       # removed: 'Respiratory Rate' is much more common
                             # Blood Pressure
                             'Arterial Blood Pressure diastolic',
                             # 'Non Invasive Blood Pressure diastolic', # removed: too many blood options, not fitting to carevue
                             # 'ART BP Diastolic',
                             'Arterial Blood Pressure systolic',
                             # 'Non Invasive Blood Pressure systolic',
                             # 'ART BP Systolic',
                             'Arterial Blood Pressure mean',
                             # 'Non Invasive Blood Pressure mean',
                             # 'ART BP Mean',
                             # O2
                             # 'Arterial O2 Saturation',
                             'O2 saturation pulseoxymetry',  # pulseoxymetry is the most common for O2
                             # 'ART %O2 saturation (PA Line)',
                             # Gauges (important for stroke use-case)
                             '14 Gauge', '16 Gauge', '18 Gauge', '20 Gauge', '22 Gauge',
                             'day_on_icu', 'oasis', 'oasis_prob'
                             # oasis will not be at the original position with the other general_info, but they must be with vitals_df because hourly/daily values
                             ]

    carevue_features: dict = {'NBP [Diastolic]': 'Arterial Blood Pressure diastolic',
                              'NBP [Systolic]': 'Arterial Blood Pressure systolic',
                              'NBP Mean': 'Arterial Blood Pressure mean',
                              'Calcium': 'Calcium Total',
                              'Chloride': 'Chloride (whole blood)',
                              'Glucose': 'Glucose (whole blood)',
                              'O2 %': 'O2 saturation pulseoxymetry',  # not ideal mapping?
                              'RBC': 'Packed Red Blood Cells',
                              'Potassium': 'Potassium (whole blood)',
                              # 'Thrombin': 'Prothrombin time',     # alternative: Bleeding Time -> there is no good match available
                              'Sodium': 'Sodium (whole blood)'}

    print('CHECK: Count of general_features:', len(general_features))
    print('CHECK: Count of vitals_features:', len(vitals_features))
    selected_features: list = general_features.copy()
    selected_features.extend(vitals_features)
    print('CHECK: Count of selected_features:', len(selected_features))

    for file in os.listdir(f'{project_path}/exports/{use_case_name}/raw/'):
        if isfile(f'{project_path}/exports/{use_case_name}/raw/{file}') and not file == '0_patient_cohort.csv':
            # load each patient: import raw_.csv (not memory-ideal because pd. has to guess each column dtype, but it works fine)
            patient_df = pd.read_csv(f'{project_path}/exports/{use_case_name}/raw/{file}', low_memory=False)
            patient_df.index.name = 'row_id'

            # keep all general features
            general_patient_df = patient_df[patient_df.columns[patient_df.columns.isin(general_features)]].copy()
            # keep all existing vitals features
            vitals_patient_df = patient_df[patient_df.columns[patient_df.columns.isin(vitals_features)]].copy()

            # Carevue: keep columns that have matching/corresponding metavision feature
            carevue_patient_df = patient_df[patient_df.columns[patient_df.columns.isin(carevue_features.keys())]].copy()
            # Carevue: rename columns (alternative to hardcoded would have been Excel mapping table)
            carevue_patient_df = carevue_patient_df.rename(columns=carevue_features)
            # Carevue: add columns to final_vitals_df
            final_vitals_df = vitals_patient_df.copy()
            for renamed_feature in carevue_patient_df.columns:
                final_vitals_df[renamed_feature] = carevue_patient_df[renamed_feature]              # insert complete column

            # Add missing vitals features with empty column
            for feature in vitals_features:
                if feature not in final_vitals_df:
                    final_vitals_df.insert(loc=0, column=feature, value=np.nan)  # loc was left out because alphabetical ordering later

            # Sum of all Gauge Types into gauges_total
            final_vitals_df.insert(loc=0, column='gauges_total', value=final_vitals_df[['22 Gauge', '20 Gauge', '18 Gauge', '16 Gauge', '14 Gauge']].sum(axis=1))
            final_vitals_df.replace(to_replace=0, value='', inplace=True)             # use empty values instead of 0 -> better mean
            final_vitals_df = final_vitals_df.drop(columns=['22 Gauge', '20 Gauge', '18 Gauge', '16 Gauge', '14 Gauge'])

            # Alphabetical Order
            final_vitals_df = final_vitals_df.reindex(sorted(final_vitals_df.columns), axis=1)
            temp_cols = final_vitals_df.columns.tolist()
            # move charttime to front
            new_cols = temp_cols[-1:] + temp_cols[:-1]
            final_vitals_df = final_vitals_df[new_cols]
            # move day_on_icu, oasis, oasis_prob to back
            final_vitals_df = final_vitals_df[
                [column for column in final_vitals_df if column not in ['day_on_icu', 'oasis', 'oasis_prob']] + [
                    'day_on_icu', 'oasis', 'oasis_prob']]

            # Add general patient info to the back
            # final_patient_df = pd.concat([vitals_patient_df, general_patient_df])       # concat not possible, needs a key for join
            final_patient_df = final_vitals_df.copy()
            for feature in general_features:
                current_columns_count = len(final_patient_df.columns)
                final_patient_df.insert(loc=current_columns_count, column=feature,
                                        value=general_patient_df[feature].iloc[0])         # only insert first row value because general_patient_info

            # export final .csv file
            filename_id = file[15:21]
            filename_string: str = f'{project_path}exports/{use_case_name}/selected_features/icustay_id_{filename_id}.csv'
            filename = filename_string.encode()
            with open(filename, 'w', newline='') as output_file:
                final_patient_df.to_csv(output_file)

    print('STATUS: Finished export_final_dataset.')
    return None


def plot_occurrence_of_features(project_path, use_case_name):
    """
    This function was an old version to select features based on their occurrence.
    It can be used to examine the available features

    It does work properly however, the features proposed by prev research are more important.
    And they, combined with general patient_info, already make up more than 50 features.

    --> occurrence of features leads to too many features.

    :param project_path:
    :param use_case_name:
    :param patient_list: list of all patients (from .csv files)
    :return: None
    """
    # IMPORT
    raw_feature_count_dict: dict = {}
    patient_count: int = 0
    for file in os.listdir(f'{project_path}/exports/{use_case_name}/raw/'):
        if isfile(f'{project_path}/exports/{use_case_name}/raw/{file}') and not file == '0_patient_cohort.csv':
            patient_count += 1
            header = pd.read_csv(f'{project_path}/exports/{use_case_name}/raw/{file}', nrows=0)  # only get header

            for feature in header:
                if feature in raw_feature_count_dict.keys():
                    raw_feature_count_dict[feature] += 1
                else:
                    raw_feature_count_dict[feature] = 1  # create with value 1

    # SORTING
    sorted_feature_count_list: list = sorted(raw_feature_count_dict.items(),
                                             key=lambda x: x[1],
                                             reverse=True)  # sorted() creates a list of tuples(), reverse = descending
    sorted_feature_count_dict: dict = dict(sorted_feature_count_list)
    # ('CHECK: The following dictionary offers an overview of the occurrence of features in the dataset. '
    #  'It can be used to select relevant features for the following analysis.')
    # print(sorted_feature_count_dict)
    print(f'Count of patients: {patient_count}, '
          f'Count of features with occurrence for all patients: {countOf(sorted_feature_count_dict.values(), patient_count)}, '
          f'Count of all available features: {len(sorted_feature_count_dict)} ')

    # FILTERING
    # manually remove features that are general patient info (will be added again later)
    general_features: list = list(
        pd.read_csv(f'{project_path}/exports/{use_case_name}/0_patient_cohort.csv', nrows=0).columns)
    general_features_complete: list = ['icustay_id', 'hadm_id', 'subject_id', 'intime', 'outtime', 'los_hours',
                                       'day_on_icu', 'icustays_count', 'age', 'patientweight', 'gender', 'ethnicity',
                                       'admission_type', 'discharge_location', 'insurance', 'language', 'religion',
                                       'marital_status',
                                       'diagnosis_text', 'dob', 'dod', 'death_in_hosp', 'death_3_days', 'death_30_days',
                                       'death_180_days',
                                       'death_365_days', 'oasis', 'oasis_prob', 'preiculos', 'gcs', 'mechvent',
                                       'electivesurgery',
                                       'stroke_type', 'hypertension_flag', 'diabetes_flag', 'cancer_flag',
                                       'obesity_flag',
                                       'drug_abuse_flag', 'sepsis_flag', 'icd9_code', 'all_icd9_codes']
    for feature in general_features_complete:
        if not feature in general_features:
            general_features.append(feature)

    # manually remove features that are known and important from prev research (will be added again later)
    important_features: list = ['charttime',
                                # novel nomogram paper:
                                'Anion gap', 'Anion Gap', 'Bicarbonate',
                                'Chloride (whole blood)', 'Calcium Total', 'Creatinine', 'Glucose (whole blood)',
                                'Potassium (whole blood)', 'Sodium (whole blood)', 'Hemoglobin', 'WBC',
                                'White Blood Cells', 'Packed Red Blood Cells', 'Platelets',
                                'Platelet Count', 'Prothrombin time', 'INR', 'Heart Rate', 'Respiratory Rate',
                                'Respiratory Rate (Total)',
                                # Blood Pressure
                                'Arterial Blood Pressure diastolic', 'Non Invasive Blood Pressure diastolic',
                                'ART BP Diastolic',
                                'Arterial Blood Pressure systolic', 'Non Invasive Blood Pressure systolic',
                                'ART BP Systolic',
                                'Arterial Blood Pressure mean', 'Non Invasive Blood Pressure mean', 'ART BP Mean',
                                'Arterial O2 Saturation', 'O2 saturation pulseoxymetry', 'ART %O2 saturation (PA Line)',
                                # Gauges
                                '20 Gauge', '18 Gauge', '22 Gauge', '16 Gauge', '14 Gauge']
    # manually remove features that can be safely dismissed
    # the following are removed, even though little medical knowledge exists
    unimportant_features: list = ['Gender', 'Religion', 'Language',  # are in general as 'gender' and 'religion'
                                  'Nasal Swab', 'Abdominal Assessment', 'Activity', 'Activity Tolerance',
                                  'Admission Weight (Kg)', 'Alarms On', 'Ambulatory aid',
                                  'Anti Embolic Device', 'Anti Embolic Device Status', 'Assistance Device',
                                  'Bowel Sounds', 'Braden Activity', 'Braden Friction/Shea',
                                  'Braden Mobility', 'Braden Moisture', 'Braden Nutrition', 'Braden Sensory Perception',
                                  'BUN', 'Chloride (serum)', 'Glucose (serum)', 'HCO3 (serum)', 'Hematocrit (serum)',
                                  'Potassium (serum)', 'Calcium non-ionized',
                                  'Cough Effort', 'Dorsal PedPulse L',
                                  'Dorsal PedPulse R', 'Ectopy Type 1', 'Edema Location', 'Education Barrier',
                                  'Education Learner', 'Education Method', 'Education Readiness',
                                  'Education Response', 'Education Topic', 'Eye Care', 'GCS - Eye Opening',
                                  'GCS - Motor Response', 'GCS - Verbal Response', 'Head of Bed',
                                  'IV/Saline lock', 'LLL Lung Sounds', 'LUL Lung Sounds', 'Mental status', 'Oral Care',
                                  'Oral Cavity', 'Pain Assessment Method', 'Pain Level Acceptable',
                                  'Pain Location', 'Pain Present', 'Pain Type', 'Parameters Checked', 'Position',
                                  'PostTib. Pulses L', 'PostTib. Pulses R',
                                  'Back Care', 'Capillary Refill L', 'Capillary Refill R', 'Cough Reflex', 'Cough Type',
                                  'Edema Amount', 'Flatus', 'LL Strength/Movement',
                                  'LU Strength/Movement', 'Nares L', 'Nares R', 'NBP Alarm Source', 'Orientation',
                                  'Pain Cause', 'Pain Management', 'Response',
                                  'RL Strength/Movement', 'RU Strength/Movement', 'Skin Care', 'Spontaneous Movement',
                                  'Admission Weight (lbs.)', 'Bed Bath',
                                  'Daily Wake Up Deferred', 'Daily Weight',
                                  'Intravenous  / IV access prior to admission', 'Pressure Reducing Device', 'Solution',
                                  'Skin Color', 'Skin Condition', 'Skin Integrity', 'Skin Temperature',
                                  'Smoking Cessation Info Offered through BIDMC Inpatient Guide',
                                  '20 Gauge Dressing Occlusive', '20 Gauge placed in outside facility',
                                  '20 Gauge Site Appear',
                                  'LLE Color', 'LLE Temp', 'Radial Pulse L', 'Radial Pulse R', 'RLE Color', 'RLE Temp',
                                  'Temperature Site', 'Therapeutic Bed', 'Turn', 'Untoward Effect',
                                  'Urea Nitrogen', 'Urine Appearance', 'Urine Color', 'Urine Source',
                                  'Respiratory Pattern', 'RLL Lung Sounds', 'RUL Lung Sounds',
                                  '18 Gauge Dressing Occlusive', '18 Gauge placed in outside facility',
                                  '18 Gauge Site Appear',
                                  'All Medications Tolerated without Adverse Side Effects', 'ALT',
                                  'Asparate Aminotransferase (AST)', 'AST',
                                  'Creatine Kinase MB Isoenzyme', 'Is the spokesperson the Health Care Proxy',
                                  'LUE Color', 'LUE Temp',
                                  'RUE Color', 'RUE Temp', 'Side Rails',
                                  'Stool Color', 'Stool Consistency', 'LLE Sensation', 'LUE Sensation', 'RLE Sensation',
                                  'RUE Sensation', 'ABP Alarm Source', 'Arterial Base Excess',
                                  'Impaired Skin Site #1', 'Sputum Color', 'Sputum Consistency',
                                  'Sputum Source', 'Stool Estimate', 'Admit from', 'Code Status', 'Service',
                                  '20 Gauge placed in the field', 'Blood Cultured', 'Currently experiencing pain',
                                  'ETT Location', 'ETT Mark (cm)', 'ETT Size (ID)', 'Expiratory Ratio', 'Flow Pattern',
                                  'Flow Rate (variable/fixed)', 'Flow Sensitivity', 'GI #1 Intub Site',
                                  'GI #1 Tube Place Check', 'GI #1 Tube Place Method', 'GI #1 Tube Status',
                                  'GI #1 Tube Type',
                                  'History of slips / falls', 'Humidification', 'Impaired Skin Drainage #1',
                                  'Incision Appearance #1',
                                  'Incision Site #1', 'Inspiratory Ratio', 'Inspiratory Time',
                                  'Known difficult intubation',
                                  'Minute Volume', 'Minute Volume Alarm - High', 'Minute Volume Alarm - Low',
                                  'Paw High', 'Peak Insp. Pressure', 'PEEP set', 'PH (dipstick)', 'Plateau Pressure',
                                  'PO Intake',
                                  'Position Change', 'Recreational drug use', 'Restraint (Non-violent)',
                                  'Self ADL', 'Slope', 'Special diet', 'Specific Gravity', 'Specific Gravity (urine)',
                                  'Spont RR',
                                  'Spont Vt', 'Subglottal Suctioning', 'Teaching directed toward',
                                  'Tidal Volume (observed)', 'Tidal Volume (set)',
                                  'Tidal Volume (spontaneous)', 'Total PEEP Level', 'Urine Culture',
                                  'Use of assistive devices', 'Ventilator Mode',
                                  'Ventilator Tank #1', 'Ventilator Tank #2', 'Ventilator Type', 'Vti High', 'Yeast',
                                  'Surrounding Tissue #1', 'Gait/Transferring', 'History of falling (within 3 mnths)'
                                  ]

    filtered_features_count_list: list = []
    SELECTED_FEATURE_THRESHOLD = 0.7
    for feature_value_tuple in sorted_feature_count_list:
        if not feature_value_tuple[0] in general_features \
                and not feature_value_tuple[0] in important_features \
                and not feature_value_tuple[0] in unimportant_features:
            if (feature_value_tuple[
                    1] / patient_count) >= SELECTED_FEATURE_THRESHOLD:  # Important: some uncommon features might actually be very important
                filtered_features_count_list.append(feature_value_tuple)
    filtered_features_count_dict: dict = dict(filtered_features_count_list)
    # print(f'CHECK filtered_features_count_dict: {filtered_features_count_dict}')
    print(f'Count of general_information features: {len(general_features)}')
    print(f'Count of important_features features: {len(important_features)}')
    print(
        f'Count of all remaining features depending on occurrence over {SELECTED_FEATURE_THRESHOLD * 100} %: {len(filtered_features_count_list)}')
    filtered_features_list = []
    for feature_tuple in filtered_features_count_list:
        filtered_features_list.append(feature_tuple[0])
    # print(f'CHECK filtered_features_count_dict: {filtered_features_list}')

    # Plot the filtered features that occur for all patients
    feature_name, feature_count = zip(
        *filtered_features_count_list)  # if only print most important features: list[:(countOf(filtered_features_count_dict.values(), patient_count) + 5)]
    plt.plot(feature_name, feature_count)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Add again the general_features and important_features to final feature-list
    selected_features: list = general_features  # 42 features
    selected_features.extend(important_features)  # 38 features
    selected_features.extend(filtered_features_list)  # approx. 48 features

    # print(selected_features)
    print(f'Count of final selected features: {len(selected_features)}')
    print('STATUS: Finished plot_occurrence_of_features.')

    return None
