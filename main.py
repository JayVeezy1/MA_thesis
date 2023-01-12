from step_1_setup_data import select_relevant_features, mimic_to_csv
from supplements import selection_icd9_codes

####### MAIN #######
if __name__ == '__main__':
    PROJECT_PATH: str = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'          # this variable must be fitted to the users local project folder
    USE_CASE_NAME: str = 'stroke_patients_data'

    ### Mimic to CSV Export
    # Step 0) Setup when first time using db:
    # mimic_to_csv.setup_postgre_files()                 # setup all needed background functions and views for postgre. Warning: Sometimes this setup from Python does not work. Then you simply copy&paste each SQL Script into PostGre QueryTool and execute it.
    # mimic_to_csv.create_table_all_diagnoses()          # create a necessary table 'all_diagnoses' where for each admission all available diagnoses are saved in the new field 'all_icd_codes' (takes approx. 45 min)
    # mimic_to_csv.create_supplement_dictionaries()      # create supplementary dictionary files
    # mimic_to_csv.load_comorbidities_into_db()          # create the necessary table 'comorbidity_codes' where the icd9_codes that are used to find important comorbidities are loaded into the DB

    # Step 1.1) Export the raw patient data for the specified use_case (icd_list) into .csv files, all available features will be exported
    mimic_to_csv.export_patients_to_csv(project_path=PROJECT_PATH,
                                        use_case_icd_list=selection_icd9_codes.icd9_00_stroke_selected,
                                        use_case_itemids=[],
                                        use_case_name=USE_CASE_NAME)

    # Step 1.2) Filter final patient.csvs for relevant features and export as 'final_dataset'
    # select_relevant_features.export_final_dataset(project_path=PROJECT_PATH,
    #                                            use_case_name=USE_CASE_NAME)

#### Upcoming TODOS
# TODO: Export ALL patients to raw_csv and filter to final_csv


#### Long Term #########################################################################################################
### CSV Import & Preprocessing
# Step 2.1) Import all .csv files as a 'Patient' Object with a related dataframe
    # -> use step_2_preprocessing.import_csv_to_patients for this

# Step 2.2) Calculate avg, min, max for each feature for each patient
# use feature_preprocessing_table.xlsx for selection of continuous vs categorical
# Depending on missing value-rate either interpolate or remove feature ?

# feature-classification-table (later for overview-analysis):
# label | classification (existing values of this feature) | patients_in_training_set (count/occurrence) | correlation_to_death | p-value (continuous) | chi-squared-value (categorical)

# Step 2.3) Combine all selected patients into a 'set' object, save this object as .pickle (with unique hash-name)


### Data Analysis
# Step 3.1) General Statistics

# Step 3.2) Correlations, Clustering, etc.


### Machine Learning Predictions
# Step 4.1) Random Forest

# Step 4.2) XGBoost

# Step 4.3) Deep Learning/Neural Network


### Fairness Metrics
# Step 5.1) Calculate Fairness for manual Subgroups


### Automated Subgroup detection
# Step 6.1) Calculate automated Subgroups and related fairness metrics

# Step 6.2) Include ASDF-Dashboard as frontend
