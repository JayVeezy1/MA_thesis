import datetime
import os
from os.path import isfile, join
import pickle
import pandas as pd

from objects.patients import Patient


def save_csvs_into_cache(project_path: str, features_df, use_case_name=None):
    """
    This function loads all previously created 'patient.csvs' where for each icustay_id all available
    features have been exported from the postgres DB into the csv files.
    This will reduce the loading time of all patients for the future analysis as the program does not need to open
    each of the .csv files but only one .pickle file.

    :param features_df: excel sheet of selected features
    :param project_path: local directory where files are stored, must be defined by the user
    :param use_case_name: the title for this use-case e.g. stroke, heart_failure, sepsis, only used to find the folder
    """
    if project_path is None:
        print('ERROR: project_path must be defined by the user.')
        return None
    if use_case_name is None:
        use_case_name = 'no_use_case_name'
        print('NOTICE: No use-case name was chosen. Export files will be saved in folder "no_use_case_name".')

    # Load .csvs into Patient class
    directory: str = project_path + 'exports/' + use_case_name + '/selected_features/'  # not very pretty
    patient_counter: int = 0
    start_time = datetime.datetime.now()

    for patient_file in os.listdir(directory):
        if isfile(join(directory, patient_file)):  # making sure only files, no other dictionaries inside the SQL folder
            patient_data = pd.read_csv(join(directory, patient_file), low_memory=False)
            temp_patient: Patient = Patient(patient_id=patient_file[11:-4],
                                            patient_data=patient_data,
                                            features_df=features_df)      # create a new Patient Obj with this
            patient_counter += 1
    print(f'CHECK: Count of .csv files that were loaded into the Patient class: {patient_counter}')

    # Save Patient class as Pickle Cache
    # Might use unique hash name for pickle file depending on selected patients
    # do we really ever need this? The 'standard' pickle should really be fast enough.
    pickle.dump({'all_patients': Patient.all_patient_objs_set},
                open(project_path + 'exports/' + use_case_name + '/complete_patients_cache.p', 'wb'))
    print(f'CHECK: Count of patients that were saved into the cache file: {len(Patient.all_patient_objs_set)}')
    print('CHECK: Took', datetime.datetime.now() - start_time, 'to execute save_csvs_into_cache.')


def load_data_from_cache(project_path, features_df, use_case_name, delete_existing_cache):
    cache_file_path = project_path + 'exports/' + use_case_name + '/complete_patients_cache.p'

    if delete_existing_cache:
        try:
            os.remove(cache_file_path)
        except FileNotFoundError:
            pass

    if not os.path.isfile(cache_file_path):
        print(f'STATUS: Creating new cache for the .csv files inside load_patients_from_cache.')
        save_csvs_into_cache(project_path=project_path, features_df=features_df, use_case_name=use_case_name)

    print(f'STATUS: Loading patient data from pickle cache {cache_file_path}')
    start_time = datetime.datetime.now()
    cache_data = pickle.load(open(cache_file_path, 'rb'))

    for patient_obj in cache_data['all_patients']:
        try:
            Patient.add_patient_obj(patient_obj)
        except KeyError:
            pass
    print('CHECK: Took', datetime.datetime.now() - start_time, 'to execute load_patients_from_cache for', len(Patient.all_patient_ids_set), 'patients.')
