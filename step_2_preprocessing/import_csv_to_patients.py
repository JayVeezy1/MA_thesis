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

    # USE THE PATIENT CLASS FOR THIS
    # todo: create the patient object with a dataframe (also with the respective raw_columns?)
    # df_patient_data = pd.read_csv(os.path.join(data_set_path, patient_filename), sep='|')
    # patient = Patient(os.path.splitext(patient_filename)[0], df_patient_data)