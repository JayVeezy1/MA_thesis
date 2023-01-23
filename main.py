from objects.patients import Patient
from step_1_setup_data import cache_IO, mimic_to_csv, select_relevant_features
from step_3_data_analysis import correlations, classification, clustering, general_statistics, data_visualization
from supplements import selection_icd9_codes

####### MAIN #######
if __name__ == '__main__':
    PROJECT_PATH: str = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'  # this variable must be fitted to the users local project folder
    USE_CASE_NAME: str = 'stroke_patients_data'

    ### Setup, MIMIC-III Export from DB, Python-Cache
    # Step 0) Setup when first time using db:
    # mimic_to_csv.setup_postgre_files()                 # setup all needed background functions and views for postgre. Warning: Sometimes this setup from Python does not work. Then you simply copy&paste each SQL Script into PostGre QueryTool and execute it.
    # mimic_to_csv.create_table_all_diagnoses()          # create a necessary table 'all_diagnoses' where for each admission all available diagnoses are saved in the new field 'all_icd_codes' (takes approx. 45 min)
    # mimic_to_csv.create_supplement_dictionaries()      # create supplementary dictionary files
    # mimic_to_csv.load_comorbidities_into_db()          # create the necessary table 'comorbidity_codes' where the icd9_codes that are used to find important comorbidities are loaded into the DB

    # Step 1.1) Export the raw patient data for the specified use_case (icd_list) into .csv files, all available features will be exported
    # stroke use-case has 1232 patients, each takes approx. 30 seconds -> 500 Minutes, 8,5 hours

    # mimic_to_csv.export_patients_to_csv(project_path=PROJECT_PATH,
    #                                     use_case_icd_list=selection_icd9_codes.icd9_00_stroke_selected,
    #                                     use_case_itemids=[],
    #                                     use_case_name=USE_CASE_NAME)

    # Step 1.2) Filter final patient.csvs for relevant features and export as 'final_dataset'
    # select_relevant_features.export_final_dataset(project_path=PROJECT_PATH, use_case_name=USE_CASE_NAME)

    # Step 1.3) Load all .csv files as a 'Patient' Object, use Pickle for Cache
    # cache_IO.save_csvs_into_cache(project_path=PROJECT_PATH, use_case_name=USE_CASE_NAME)
    cache_IO.load_patients_from_cache(project_path=PROJECT_PATH,
                                      use_case_name=USE_CASE_NAME,
                                      delete_existing_cache=False)

    ### Preprocessing
    # Step 2.1) Calculate avg, min, max for each feature for each patient
    complete_avg_patient_cohort = Patient.get_avg_patient_cohort(project_path=PROJECT_PATH,
                                                                 use_case_name=USE_CASE_NAME,
                                                                 selected_patients=[])  # if empty -> all

    SELECTED_FEATURES = Patient.feature_categories[Patient.feature_categories['selected_for_analysis'] == 'yes'][
        'feature_name'].to_list()
    avg_hemorrhage_cohort = complete_avg_patient_cohort[complete_avg_patient_cohort['stroke_type'] == 'hemorrhagic']
    avg_ischemic_cohort = complete_avg_patient_cohort[complete_avg_patient_cohort['stroke_type'] == 'ischemic']

    # Step 2.2) Impute, Interpolate, Normalize dataframes for each patient
    # todo 0: work on NaN, interpolation, also min/max-columns for features?

    #### Long Term #########################################################################################################
    ### Data Analysis
    # Step 3.1) General Statistics
    general_statistics.calculate_deaths_table(avg_patient_cohort=complete_avg_patient_cohort,
                                              cohort_title='complete_avg_patient_cohort',
                                              selected_features=SELECTED_FEATURES,
                                              save_to_file=True)

    # todo 3: finish 'feature_classification_table'
    # label | classification (existing values of this feature) | patients_in_training_set (count/occurrence) | correlation_to_death | p-value (continuous) | chi-squared-value (categorical) | NaN Amount
    # general_statistics.calculate_feature_overview_table(avg_patient_cohort=complete_avg_patient_cohort,
    #                                          cohort_title='complete_avg_patient_cohort',
    #                                        selected_features=SELECTED_FEATURES,
    #                                                save_to_file=True)

    # Step 3.2) Correlations, Clustering, etc.
    # data_visualization.display_pacmap(avg_patient_cohort=avg_hemorrhage_cohort,
    #                                cohort_title='avg_hemorrhage_cohort',
    #                               selected_features=SELECTED_FEATURES,
    #                              selected_dependent_variable='death_in_hosp',
    #                             save_to_file=True)

    # Correlation Prototype
    # correlations.calculate_correlations_on_cohort(avg_patient_cohort=avg_hemorrhage_cohort,
    #                                            cohort_title='avg_hemorrhage_cohort',
    #                                           selected_features=SELECTED_FEATURES,
    #                                          selected_dependent_variable='death_in_hosp',         # death_3_days, death_in_hosp
    #                                          save_to_file=True)

    # todo 4: Finish working on Clustering Prototype
    # clustering.calculate_k_means_on_cohort(avg_patient_cohort=avg_hemorrhage_cohort,
    #                                      cohort_title='avg_hemorrhage_cohort',
    #                                     selected_features=SELECTED_FEATURES,
    #                                    selected_dependent_variable='death_in_hosp',
    #                                    save_to_file=True)

### Machine Learning Predictions
# Step 4.1) Random Forest
# todo 5: Check again Prediction Prototype
# classification.calculate_RF_on_cohort(avg_patient_cohort=avg_hemorrhage_cohort,
#                                     cohort_title='avg_hemorrhage_cohort',
#                                    selected_features=SELECTED_FEATURES,
#                                   selected_dependent_variable='death_in_hosp')

# Step 4.2) XGBoost

# Step 4.3) Deep Learning/Neural Network


### Fairness Metrics
# Step 5.1) Calculate Fairness for manual Subgroups


### Automated Subgroup detection
# Step 6.1) Calculate automated Subgroups and related fairness metrics

# Step 6.2) Include ASDF-Dashboard as frontend
