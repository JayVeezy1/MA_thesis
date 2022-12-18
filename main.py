import mimic_to_csv
import supplements.icd9_codes

####### MAIN #######
if __name__ == '__main__':
    ### Mimic to CSV Export
    # Step 0) Only first time using db: create the needed temp table 'all_diagnoses_icd' (takes approx. 45 min):
    # mimic_to_csv.create_table_all_diagnoses()

    # Step 1.1) Export the patient_cohort for the specified use_case (icd_list) into .csv file:
    mimic_to_csv.export_patient_cohort_to_csv(use_case_icd_list=supplements.icd9_codes.icd9_00_stroke_selected,
                                              use_case_name='testing_stroke')

# INFO: ICD9 Codes for Stroke found: 1447
# INFO: Total available patients after filtering like research: 13762 (which they also had -> correct)


#### Next Task #########################################################################################################
    # TODO: Step 1.2) Export a unique .csv file with Chart-Data (time series) for each admission (from now on admission = patient)


#### Long Term #########################################################################################################
    ### CSV Import & Preprocessing
    # Step 2.1) Import all .csv files as a 'Patient' Object with a related dataframe

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

