import pandas as pd
import numpy as np
from pandas.core.interchange import dataframe


class NotUniqueIDError(Exception):
    # is this really necessary?
    pass


class Patient:
    """
    Class containing all patients that are imported from the .csv files.

    Note: The .csv files/ids are actually based on icustays. A real patient might have had multiple icustays,
    but only their first icustay will be included in this analysis. Meaning the icustay_id will be used as 'patient_id'.
    """

    # CLASS FEATURES
    all_patient_ids_set: set = set()                  # not really needed, could be derived with a function get_ids_from_objs_set
    all_patient_objs_set: set = set()                 # keys from the cache file should be patients_ids
    feature_categories: dataframe = pd.read_excel('./supplements/feature_preprocessing_table.xlsx')

    def __init__(self, patient_id: str, patient_data: dataframe):
        self.features: list = list(patient_data.columns.values)

        if patient_id not in Patient.all_patient_ids_set:
            self.patient_id = patient_id
            Patient.all_patient_ids_set.add(patient_id)
            Patient.all_patient_objs_set.add(self)
        else:
            self.features = None
            self.raw_data = None
            self.patient_id = None
            print(f'CHECK: Patient {patient_id} already exists.')
            # simply not add this ID to all_patient_ids, much easier than throwing of an exception
            # raise NotUniqueIDError(f'Icustay ID {patient_id} already exists in all_patients_set')

        # patient related datasets
        self.raw_data: dataframe = self.get_clean_raw_data(patient_data)  # raw = timeseries & no imputation/interpolation
        self.imputed_data: dataframe = self.get_imputed_data()
        self.interpolated_data: dataframe = self.get_interpolated_data()  # interpolated built upon imputed?
        self.normalized_data: dataframe = self.get_normalized_data()  # normalized = z-values
        self.avg_data: dataframe = self.get_avg_data()  # avg built upon interpolated?


    def __del__(self):
        if self.patient_id in Patient.all_patient_ids_set:
            Patient.all_patient_ids_set.remove(self.patient_id)
        if self in Patient.all_patient_objs_set:
            Patient.all_patient_objs_set.remove(self)

    def get_clean_raw_data(self, patient_data: dataframe) -> dataframe:
        # patient_data = patient_data.fillna(0)            # todo: check/ask if that is needed?
        # also fill empty columns with a value?
        cleaned_raw_data: dataframe = patient_data.copy()

        # Preprocess non-numeric columns
        cleaned_raw_data['gender'] = np.where(cleaned_raw_data['gender'] == 'F', 0, 1)  # F = 0, M = 1

        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'WHITE - RUSSIAN', 'ethnicity'] = 'WHITE'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'WHITE - BRAZILIAN', 'ethnicity'] = 'WHITE'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'WHITE - OTHER EUROPEAN', 'ethnicity'] = 'WHITE'

        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'UNABLE TO OBTAIN', 'ethnicity'] = 'UNKNOWN/NOT SPECIFIED'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'PATIENT DECLINED TO ANSWER', 'ethnicity'] = 'UNKNOWN/NOT SPECIFIED'

        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'MULTI RACE ETHNICITY', 'ethnicity'] = 'OTHER'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'MIDDLE EASTERN', 'ethnicity'] = 'OTHER'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'AMERICAN INDIAN/ALASKA NATIVE', 'ethnicity'] = 'OTHER'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'PORTUGUESE', 'ethnicity'] = 'OTHER'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER', 'ethnicity'] = 'OTHER'

        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'BLACK/AFRICAN AMERICAN', 'ethnicity'] = 'BLACK'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'BLACK/AFRICAN', 'ethnicity'] = 'BLACK'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'BLACK/HAITIAN', 'ethnicity'] = 'BLACK'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'BLACK/CAPE VERDEAN', 'ethnicity'] = 'BLACK'

        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'HISPANIC/LATINO - PUERTO RICAN', 'ethnicity'] = 'HISPANIC OR LATINO'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'HISPANIC/LATINO - DOMINICAN', 'ethnicity'] = 'HISPANIC OR LATINO'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'HISPANIC/LATINO - CUBAN', 'ethnicity'] = 'HISPANIC OR LATINO'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'HISPANIC/LATINO - SALVADORAN', 'ethnicity'] = 'HISPANIC OR LATINO'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'HISPANIC/LATINO - GUATEMALAN', 'ethnicity'] = 'HISPANIC OR LATINO'

        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - CHINESE', 'ethnicity'] = 'ASIAN'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - CAMBODIAN', 'ethnicity'] = 'ASIAN'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - OTHER', 'ethnicity'] = 'ASIAN'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - VIETNAMESE', 'ethnicity'] = 'ASIAN'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - FILIPINO', 'ethnicity'] = 'ASIAN'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - JAPANESE', 'ethnicity'] = 'ASIAN'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - ASIAN INDIAN', 'ethnicity'] = 'ASIAN'
        cleaned_raw_data.loc[cleaned_raw_data['ethnicity'] == 'ASIAN - KOREAN', 'ethnicity'] = 'ASIAN'

        # Remove 'ERROR' fields
        for feature in cleaned_raw_data.columns:
            if len(cleaned_raw_data.loc[cleaned_raw_data[feature].isin(['ERROR']), feature]) > 0:
                print(f'CHECK: Replacing "ERROR" from feature {feature} for patient {cleaned_raw_data["icustay_id"][0]}')
                cleaned_raw_data.loc[cleaned_raw_data[feature].isin(['ERROR']), feature] = np.nan
                cleaned_raw_data[feature] = cleaned_raw_data[feature].astype(float)     # is the feature contained 'ERROR', all existing values were set as string

        return cleaned_raw_data

    def get_imputed_data(self) -> dataframe:
        return self.raw_data

    def get_interpolated_data(self) -> dataframe:
        return self.raw_data

    def get_normalized_data(self) -> dataframe:
        return self.raw_data

    def get_avg_data(self) -> dataframe:
        avg_df: dataframe = pd.DataFrame()
        avg_df = avg_df.assign(DUMMY_COLUMN=[1])

        for feature in self.raw_data.columns:
            # get feature_type from supplement table 'feature_categories'
            if feature in Patient.feature_categories['feature_name'].tolist():
                feature_type = Patient.feature_categories.loc[Patient.feature_categories['feature_name'] == feature, 'categorical_or_continuous'].item()
            else:
                feature_type = 'unknown'

            if feature_type == 'none':
                # simply take first available value
                avg_df.insert(len(avg_df.columns), feature, self.raw_data[feature][0])
            elif feature_type == 'continuous':
                # calculate average over all rows
                try:
                    avg_df.insert(len(avg_df.columns), feature, round(self.raw_data[feature].mean(), 3))
                except TypeError:
                    avg_df.insert(len(avg_df.columns), feature, 'NaN')
                    print(f'CHECK: Error occurred for mean calculation of feature {feature} for patient {self.patient_id}')
            elif feature_type == 'single_value':
                # simply take first available value
                avg_df.insert(len(avg_df.columns), feature, self.raw_data[feature][0])
            elif feature_type == 'flag_value':
                # if ever there is a row with positive flag -> complete column positive = 1
                if self.raw_data[feature].sum() > 0:
                    avg_df.insert(len(avg_df.columns), feature, 1)
                else:
                    avg_df.insert(len(avg_df.columns), feature, 0)
            else:
                print(f'WARNING: Unknown column_type: {feature_type} for column {feature}')

        avg_df = avg_df.drop(columns=['DUMMY_COLUMN'])
        avg_df = avg_df.drop(columns=['row_id'])
        # print('CHECK: Average dataframe.')
        # print(avg_df)

        return avg_df


    @classmethod
    def get_patient(cls, patient_id):
        for patient in cls.all_patient_objs_set:
            if patient.patient_id == patient_id:
                return patient

    @classmethod
    def get_avg_patient_cohort(cls, project_path, use_case_name, selected_patients) -> dataframe:
        # Important: all patients must be already loaded (from cache) at this point
        if not selected_patients:
            selected_patients = Patient.all_patient_objs_set

        avg_dataframes: list = []
        for patient in selected_patients:
            avg_dataframes.append(patient.avg_data)
        avg_patient_cohort_dataframe: dataframe = pd.concat(avg_dataframes)
        avg_patient_cohort_dataframe = avg_patient_cohort_dataframe.sort_values(by=['icustay_id'], axis=0)
        avg_patient_cohort_dataframe = avg_patient_cohort_dataframe.reset_index(drop=True)
        avg_patient_cohort_dataframe.index.name = 'index'

        # Export avg_patient_cohort
        filename_string: str = f'{project_path}/exports/{use_case_name}/avg_patient_cohort.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            avg_patient_cohort_dataframe.to_csv(output_file)

        print(f'STATUS: avg_patient_cohort file was saved to {project_path}/exports/{use_case_name}')

        return avg_patient_cohort_dataframe

    @classmethod
    def add_patient_obj(cls, patient_obj):
        if patient_obj.patient_id in Patient.all_patient_ids_set:
            # print(f'CHECK: patient {patient_obj.patient_id} already exists in Patient.all_patient_objs_set.')
            pass
        else:
            # print(f'CHECK: Adding new patient {patient_obj.patient_id} to Patient.all_patient_objs_set.')
            # print(f'CHECK: Current count of Patient.all_patient_objs_set: {len(Patient.all_patient_objs_set)}')
            Patient.all_patient_ids_set.add(patient_obj.patient_id)
            Patient.all_patient_objs_set.add(patient_obj)
