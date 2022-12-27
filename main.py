import mimic_to_csv
import supplements.icd9_codes
import supplements.itemids_list_mv

####### MAIN #######
if __name__ == '__main__':
    ### Mimic to CSV Export
    # Step 0) Only first time using db: create the needed temp table 'all_diagnoses_icd' (takes approx. 45 min):
    # mimic_to_csv.create_table_all_diagnoses()

    # Step 1.1) Export the patient data for the specified use_case (icd_list) into .csv files:
    mimic_to_csv.export_patients_to_csv(use_case_icd_list=supplements.icd9_codes.icd9_00_stroke_selected,
                                        use_case_itemids=supplements.itemids_list_mv.selected_itemids_stroke,
                                        use_case_name='testing_stroke')

#### Current Tasks
    # TODO 1: add prescription events, also filter lab_events charttime < outtime
    # todo: get labevents, etc events as dictionary or better dataframe with label, itemid, valueuom (unit of measurement)

    # TODO 2: choose chart_events itemids -> from other papers?
        #  alternative: simply take all 2000 features -> one big export
        #  filter inside python after import depending on occurence of the itemids

    # langfristig
    # TODO 4: cleanup SQL Files, check why it now takes 40 seconds, previously it was only 20 seconds
    # TODO 5: save a label-to-measurement dict in supplements, also those supplements could have been automated with SQL, but they dont change so thats ok? -> explain in Readme.md

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

