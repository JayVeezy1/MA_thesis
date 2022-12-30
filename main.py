from mimic_to_csv_folder import mimic_to_csv
import supplements.selection_icd9_codes
import supplements.selection_of_features

####### MAIN #######
if __name__ == '__main__':
    project_path: str = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'          # this variable must be fitted to the users local project folder
    ### Mimic to CSV Export
    # Step 0) Setup when first time using db:
    # mimic_to_csv.setup_postgre_files()                 # setup all needed background functions and views for postgre. Warning: Sometimes this setup from Python does not work. Then you simply copy&paste each SQL Script into PostGre QueryTool and execute it.
    # mimic_to_csv.create_label_dictionaries()             # create supplementary dictionary files
    # mimic_to_csv.create_table_all_diagnoses()          # create a necessary table 'all_diagnoses' where for each admission all available diagnoses are saved in the new field 'all_icd_codes' (takes approx. 45 min)

    # Step 1.1) Export the patient data for the specified use_case (icd_list) into .csv files:
    mimic_to_csv.export_patients_to_csv(project_path=project_path,
                                        use_case_icd_list=supplements.selection_icd9_codes.icd9_00_stroke_selected,
                                        use_case_itemids=[],
                                        use_case_name='testing_stroke_no_selected_labels')

#### Current Tasks
# todo 0: export dictionaries again, event dictionary still has , or ; in label

# TODO 1: OASIS derive scoring systems (oasis) inside SQL based on code in github
    # https://github.com/caisr-hh/Dayly-SAPS-III-and-OASIS-scores-for-MIMIC-III/blob/master/oasis-all-day.sql
    # how does the final view look like, which of the columns do i want to join to my patient_cohort? probably: final oasis score, ventilation, surgery

# TODO 2.1: COMORBIDITIES derive comorbidities (diabetes, cancer) based on paper with icd9 codes

# TODO 2.2: SEPSIS where do i get explicit sepsis from? simply icd9-code in diagnosis? -> like a comorbidity?



#### Next Step: Export
# TODO: Export patients with ALL of their available features
    # time is the same, file-size is huge but ok

# Step 1.2) Derive most relevant features from all raw patient .csv-files
# TODO: Select features inside python after import depending on occurrence of the itemids
    # Selection: general patient info & count of features & guided by other research & needed for scoring systems
    # feature-overview-table (first step for feature-selection):
    # label | item_id | count | variable_type (categorical(only 1 row) or continuous (average, min, max)) | selected (selected_general_patient_data or selected_because_research or selected_count_over_xx) |  | removed
    # afterwards save the filtered patients again as 'final-filtered-csvs'


#### Long Term #########################################################################################################
### CSV Import & Preprocessing
# Step 2.1) Import all .csv files as a 'Patient' Object with a related dataframe

# Step 2.2) Calculate avg, min, max for each feature for each patient
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
