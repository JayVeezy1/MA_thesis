from mimic_to_csv_folder import mimic_to_csv
import supplements.selection_icd9_codes
import supplements.selection_itemids_list_mv

####### MAIN #######
if __name__ == '__main__':
    ### Mimic to CSV Export
    # Step 0) Setup when first time using db:
        # mimic_to_csv.setup_postgre_files()                 # setup all needed background functions and views for postgre. Warning: Sometimes this setup from Python does not work. Then you simply copy&paste each SQL Script into PostGre QueryTool and execute it.
        # mimic_to_csv.create_label_dictionaries()           # create supplementary dictionary files
        # mimic_to_csv.create_table_all_diagnoses()          # create a necessary table 'all_diagnoses' where for each admission all available diagnoses are saved in the new field 'all_icd_codes' (takes approx. 45 min)

    # Step 1.1) Export the patient data for the specified use_case (icd_list) into .csv files:
        mimic_to_csv.export_patients_to_csv(use_case_icd_list=supplements.selection_icd9_codes.icd9_00_stroke_selected,
                                            use_case_itemids=supplements.selection_itemids_list_mv.selected_itemids_stroke,
                                            use_case_name='testing_stroke_with_selected_labels')

#### Current Tasks
# TODO 1: compare research guidelines which icd9 codes are relevant for stroke -> then export could possibly start

# TODO 2: Export patients with ALL of their available features -> build new transposed_patient version without filtering
# first step here: check how long it will take on average for a patient to take all labels, also check estimated file-size
# probably best to take all labels into csv, then filter inside python. Also the research-guided labels is still possible inside python
# alternative: simply take all 2200 item_id features + 700 labevents features -> one big export
# filter inside python after import depending on occurrence of the itemids

    # Step 1.2) Derive most relevant features from all raw patient .csv-files
    # export final-filtered patients again as .csv files -> zwischenschritt but this will be a clear basis for the analysis
    # TODO: choose which chart_events itemids are mandatory -> from other papers?


#### Long Term #########################################################################################################
    ### CSV Import & Preprocessing
    # Step 2.1) Import all .csv files as a 'Patient' Object with a related dataframe
    # addition of important side-illnesses as features: create new columns secondary_illness_cancer, secondary_illness_diabetes, secondary_illness_hypertension depending on icd9_code_list

    # Step 2.2) Calculate avg, min, max for each feature for each patient
    # Depending on missing value-rate either interpolate or remove feature ?

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

