import csv
import os
from operator import countOf

import pandas as pd
from os.path import isfile

from matplotlib import pyplot as plt


def get_selected_features(project_path, use_case_name) -> list:
    """
    This function selects the features that will be used for the final_dataset

    :param project_path:
    :param use_case_name:
    :param patient_list: list of all patients (from .csv files)
    :return: None
    """
    # IMPORT
    raw_feature_count_dict: dict = {}
    patient_count: int = 0
    for file in os.listdir(f'{project_path}/exports/{use_case_name}/'):
        if isfile(f'{project_path}/exports/{use_case_name}/{file}') and not file == '0_patient_cohort.csv':
            patient_count += 1
            header = pd.read_csv(f'{project_path}/exports/{use_case_name}/{file}', nrows=0)  # only get header

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
    # manually remove features that can be safely dismissed (without medical knowledge)
    unimportant_features: list = ['Gender', 'Religion',  'Language',   # are in general as 'gender' and 'religion'
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
    SELECTED_FEATURE_THRESHOLD = 0.9
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
    print(f'Count of all remaining features depending on occurrence over {SELECTED_FEATURE_THRESHOLD*100} %: {len(filtered_features_count_list)}')
    filtered_features_list = []
    for feature_tuple in filtered_features_count_list:
        filtered_features_list.append(feature_tuple[0])
    # print(f'CHECK filtered_features_count_dict: {filtered_features_list}')

    # Plot the filtered features that occur for all patients
    feature_name, feature_count = zip(*filtered_features_count_list)          # if only print most important features: list[:(countOf(filtered_features_count_dict.values(), patient_count) + 5)]
    plt.plot(feature_name, feature_count)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # Add again the general_features and important_features to final feature-list
    selected_features: list = general_features                  # 42 features
    selected_features.extend(important_features)                # 38 features
    selected_features.extend(filtered_features_list)            # approx. 48 features

    print(selected_features)
    print(f'Count of final selected features: {len(selected_features)}')
    print('STATUS: Finished get_feature_selection.')

    # might even be useful to save selected_features with other supplements as csv

    return selected_features


def export_final_dataset(project_path, use_case_name, selected_features: list):
    # step 1: load each patient
    for file in os.listdir(f'{project_path}/exports/{use_case_name}/'):
        if isfile(f'{project_path}/exports/{use_case_name}/{file}') and not file == '0_patient_cohort.csv':
            patient_df = pd.read_csv(f'{project_path}/exports/{use_case_name}/{file}')

            filtered_patient_df = patient_df(columns=selected_features)  # todo somehow like this

            # also if patient doesnt have feature -> feature value = None

            # step 2: export the patient but filtered with selected_features (and maybe sort these too?) into a new folder

    return None
