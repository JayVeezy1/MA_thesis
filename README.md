# MA Thesis
## Part 1 - Export of Patients into .csv
**What is this:**

The first part of this project prepares the MIMIC-III dataset for data analysis. These Python scripts can be used to 
export patients (based on icustay_ids) from a postgre database. 

Thus, in a prior step, the data must have been imported and set-up in the postgres database (instructions available at https://physionet.org/content/mimiciii/1.4/).
Also, in the 'supplements/SQL' folder, necessary SQL scripts are provided, to set up further required PL/pgSQL functions. 
Once, this setup is finished a user can run all functions from the main.py Python script.

The actual data for the MIMIC-III dataset can not be shared here. It is available, upon request, through PhysioNet.

**Output:**

An overview of all selected patients is created in '0_patient_cohort.csv'. For each patient a .csv file is created
with their hourly chart-events (time-series data). 

**Estimated Time:** 

Getting access to the MIMIC-III dataset by PhysioNet takes between 1-4 weeks. 
For this, also a course regarding data-protection had to be conducted, which regularly takes about 4 hours.
The download and import of the data into the postgre database takes about 5 hours.

Running the set-up for the table 'all_diagnoses_icd' takes about 45 minutes.
There, for each admission all available diagnoses are saved in the new field 'all_icd_codes' as an array. 
Thus, in later filtering for only one diagnoses-type, all other diagnoses for this admission are not lost.
Creation of the patient_cohort can be calculated very quickly (10 seconds). However, the creation of the single patient files with 
all their chart-events takes up multiple hours. 

**Optional Filtering:** 

The patients can be filtered with the icd9-codes (diagnoses at admission). 
Also, the required labels can be chosen by the user. Some further filtering is conducted, which can not be changed by the user. This filtering was based on previous research,
for example, data with error-flags or underage patients where removed.

# WIP

The implementation of the following steps makes up the core of my master thesis.

## Part 2 - CSV Import & Preprocessing
    # Step 2.1) Import all .csv files as a 'Patient' Object with a related dataframe

    # Step 2.2) Calculate avg, min, max for each feature for each patient
    # Depending on missing value-rate either interpolate or remove feature ?

    # Step 2.3) Combine all selected patients into a 'set' object, save this object as .pickle (with unique hash-name)
## Part 3 - Data Analysis
    # Step 3.1) General Statistics

    # Step 3.2) Correlations, Clustering, etc.


## Part 4 - Machine Learning Predictions
    # Step 4.1) Random Forest

    # Step 4.2) XGBoost

    # Step 4.3) Deep Learning/Neural Network


## Part 5 - Fairness Metrics
    # Step 5.1) Calculate Fairness for manual Subgroups


## Part 6 - Automated Subgroup detection
    # Step 6.1) Calculate automated Subgroups and related fairness metrics

    # Step 6.2) Include ASDF-Dashboard as frontend