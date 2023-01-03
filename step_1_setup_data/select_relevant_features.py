def import_complete_patient_files(project_path: str, use_case_name=None) -> list | None:
    """
    This function imports the previously created 'complete_patient.csvs' where for an icustay_id all available
    features have been exported from the postgres DB into the csv files.

    :param project_path: local directory where files are stored, must be defined by the user
    :param use_case_name: the title for this use-case e.g. stroke, heart_failure, sepsis, only used to find the folder
    :return: None
    """
    if project_path is None:
        print('ERROR: project_path must be defined by the user.')
        return None
    if use_case_name is None:
        use_case_name = 'no_use_case_name'
        print('NOTICE: No use-case name was chosen. Export files will be saved in folder "no_use_case_name".')

    patient_list: list = []

    # todo: import each file after each other

    # todo: create a patient object or a dataframe?

    print('STATUS: Finished import_complete_patient_files.')

    return patient_list


def get_feature_count_dict(patient_list: list) -> None | dict:
    """
    This function imports the previously created 'complete_patient.csvs' where for a icustay_id all available
    features have been exported from the postgres DB into the csv files.

    :param patient_list: list of all patients (from .csv files)
    :return: None
    """
    if patient_list is None:
        print('ERROR: project_path must be defined by the user.')
        return None

    feature_count_dict: dict = {}
    # create a feature_dictionary to count the occurrence of each feature
    for patient in patient_list:
        for feature in patient.column_names:
            feature_count_dict[feature] += 1            # todo: add count for the feature

    # print out the feature_dictionary to manually check which features are important
    print('CHECK: The following dictionary offers an overview of the occurrence of features in the dataset.'
          'It can be used to select relevant features for the following analysis.')

    print(feature_count_dict)

    print('STATUS: Finished export_patients_to_csv.')

    return feature_count_dict


def get_feature_selection(project_path, use_case_name) -> list:
    """
    This function selects the features that will be used for the final_dataset
    :param project_path:
    :param use_case_name:
    :param patient_list: list of all patients (from .csv files)
    :return: None
    """
    # TODO 1: Test import of patient-csv files into patient-object with related dataframe
    patient_list: list = import_complete_patient_files(project_path, use_case_name)


    feature_count_dict: dict = get_feature_count_dict(patient_list)
    # used for selection:
    # count of existing features
    # important features from research (+ needed for scoring systems)
    # general patient features

    # Goal:
    # feature-overview-table (first step for feature-selection):
    # label | item_id | count | variable_type (categorical(only 1 row) or continuous (average, min, max)) | selected (selected_general_patient_data or selected_because_research or selected_count_over_xx) |  | removed

    feature_selection: list = []


    # todo: decide if print of the list or also export as .csv? might be good to save with other supplements

    return feature_selection


def export_final_dataset(feature_selection):
    return None