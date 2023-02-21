import pandas as pd
import numpy as np
from pandas.core.interchange import dataframe


class Patient:
    """
    Class containing all patients that are imported from the .csv files.

    Note: The .csv files/ids are actually based on icustays. A real patient might have had multiple icustays,
    but only their first icustay will be included in this analysis. Meaning the icustay_id will be used as 'patient_id'.
    """
    # CLASS FEATURES
    all_patient_ids_set: set = set()                  # not really needed, could be derived with a function get_ids_from_objs_set
    all_patient_objs_set: set = set()                 # keys from the cache file should be patients_ids

    def __init__(self, patient_id: str, patient_data: dataframe, features_df: dataframe):
        self.features: list = list(patient_data.columns.values)
        # todo future work: maybe move filtering of stroke_type and infarct_type from SQL Script into Patient Class -> PRO: more flexible for future use-cases instead of SQL, CON: but must also be inside cohort.csv for filtering
        # todo future work: add interpolation/imputation(depending on NaN)/outliers to timeseries -> if use timeseries for analysis

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
            # Idea: raise NotUniqueIDError(f'Icustay ID {patient_id} already exists in all_patients_set')

        # patient related datasets
        self.raw_data: dataframe = self.get_clean_raw_data(patient_data, features_df)  # raw = timeseries & no imputation/interpolation
        self.imputed_data: dataframe = self.get_imputed_data()
        self.interpolated_data: dataframe = self.get_interpolated_data()  # interpolated built upon imputed?
        self.normalized_data: dataframe = self.get_normalized_data()  # normalized = z-values
        self.avg_data: dataframe = self.get_avg_data(features_df)  # avg built upon interpolated?


    def __del__(self):
        if self.patient_id in Patient.all_patient_ids_set:
            Patient.all_patient_ids_set.remove(self.patient_id)
        if self in Patient.all_patient_objs_set:
            Patient.all_patient_objs_set.remove(self)

    def get_clean_raw_data(self, patient_data: dataframe, feature_df: dataframe) -> dataframe:
        # Transform datetime to int     -> does not work, preiculos is 'string' in format day:hour:minute, this feature is not that important
        # try:
        #     patient_data['preiculos'] = pd.to_timedelta(patient_data['preiculos'])
        #     patient_data['preiculos'] = patient_data['preiculos'].dt.total_seconds() / (24 * 60 * 60)
        #     patient_data['preiculos'].astype('int64')
        # except KeyError as e:
        #    pass

        # Remove strings from continuous columns
        numbers_data = patient_data.copy()
        continuous_features = feature_df.query('categorical_or_continuous=="continuous"')['feature_name']
        cleaned_raw_data = (numbers_data.drop(continuous_features, axis=1)                                      # first remove all continuous columns
                            .join(numbers_data[continuous_features].apply(pd.to_numeric, errors='coerce')))     # then add back with applied function
        # cleaned_raw_data = cleaned_raw_data[cleaned_raw_data[continuous_features].notnull().all(axis=1)]      # only keep columns where ALL entries are not-null

        # Preprocess non-numeric columns
        cleaned_raw_data['gender'] = np.where(numbers_data['gender'] == 'F', 0, 1)  # F = 0, M = 1

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


        # Remove known Outliers -> todo maybe: make this automatic? If variance of a value > 1000 of mean, remove?
        if cleaned_raw_data["icustay_id"][0] == int('228194'):
            print(f'CHECK: Removing outliers for patient {cleaned_raw_data["icustay_id"][0]}')
            cleaned_raw_data.loc[cleaned_raw_data['row_id'] == 198, 'O2 saturation pulseoxymetry'] = 0

        return cleaned_raw_data

    def get_imputed_data(self) -> dataframe:
        return self.raw_data

    def get_interpolated_data(self) -> dataframe:
        return self.raw_data

    def get_normalized_data(self) -> dataframe:
        return self.raw_data

    def get_avg_data(self, features_df: dataframe) -> dataframe:
        avg_df: dataframe = pd.DataFrame()
        avg_df = avg_df.assign(DUMMY_COLUMN=[1])

        for feature in self.raw_data.columns:
            # get feature_type from supplement table 'feature_categories'
            if feature in features_df['feature_name'].tolist():
                feature_type = features_df.loc[features_df['feature_name'] == feature, 'categorical_or_continuous'].item()
            else:
                feature_type = 'unknown'

            if feature_type == 'none':
                # simply take first available value
                avg_df.insert(len(avg_df.columns), feature, self.raw_data[feature][0])
            elif feature_type == 'continuous':
                # calculate average over all rows
                try:
                    avg_df.insert(len(avg_df.columns), feature, round(self.raw_data[feature].mean(), 3))        # Error should not occur because cleaning data in earlier step
                except TypeError:
                    avg_df.insert(len(avg_df.columns), feature, 'NaN')
                    print(f'CHECK: Error occurred for mean calculation of feature {feature} for patient {self.patient_id}')
            elif feature_type == 'single_value':
                # simply take first available value
                avg_df.insert(len(avg_df.columns), feature, self.raw_data[feature][0])
            elif feature_type == 'categorical':
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

        print('CHECK: Count of selected patients for get_avg_patient_cohort: ', len(selected_patients))

        avg_dataframes: list = []
        for patient in selected_patients:
            avg_dataframes.append(patient.avg_data)

        avg_patient_cohort: dataframe = pd.concat(avg_dataframes)
        avg_patient_cohort = avg_patient_cohort.sort_values(by=['icustay_id'], axis=0)
        avg_patient_cohort = avg_patient_cohort.reset_index(drop=True)

        # Export avg_patient_cohort
        filename_string: str = f'{project_path}exports/{use_case_name}/avg_patient_cohort.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:
            avg_patient_cohort.to_csv(output_file, index=False)

        print(f'CHECK: avg_patient_cohort file was saved to {project_path}exports/{use_case_name}')

        return avg_patient_cohort

    @classmethod
    def get_avg_scaled_data(cls, avg_patient_cohort) -> dataframe:
        # TODO: Choose which normalization option!
        # normalization (between 0 and 1):
        # min-max-scaling: (df - df.min()) / (df.max() - df.min())

        # scaling (between -1 and 1) options:
        # pandas (unbiased): (x-x.mean())/ x.std()
        # sklearn (biased): scaler.fit_transform()

        # only scale numbers columns
        avg_patient_cohort_num = avg_patient_cohort.select_dtypes(include='number')

        # scaling method: min-max-normalization
        scaled_avg_cohort_num = (avg_patient_cohort_num - avg_patient_cohort_num.min()) / (
                    avg_patient_cohort_num.max() - avg_patient_cohort_num.min())

        avg_patient_cohort[avg_patient_cohort_num.columns] = scaled_avg_cohort_num   # throws SettingWithCopyWarning but works as intended   # todo: check again

        # still SettingWithCopyWarning
        # for column in avg_patient_cohort_num.columns:
          #  avg_patient_cohort.loc[:, column] = scaled_avg_cohort_num.loc[:, column]

        return avg_patient_cohort


    @classmethod
    def get_NAN_for_feature_in_cohort(cls, avg_patient_cohort_dataframe: dataframe, selected_feature) -> int:
        return avg_patient_cohort_dataframe[selected_feature].isna().sum()

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

    @classmethod
    def get_avg_norm_data(self):
        pass
