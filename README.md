# MA Thesis
## Part 1 - Export of Patients into .csv
**What is this:**

The first part of this project prepares the MIMIC-III dataset for data analysis. These Python scripts can be used to 
export patients (based on icustay_ids) from a postgre database. 

Thus, in a prior step, the data must have been imported and set-up in the postgres database (instructions available at https://physionet.org/content/mimiciii/1.4/).
Also, in the 'supplements/SQL' folder, necessary SQL scripts are provided, to set up further required PL/pgSQL functions. 
Once, this setup is finished a user can run all functions from the main.py Python script.

The actual data for the MIMIC-III dataset can not be shared here. It is available, upon request, through PhysioNet.

**Analysis Goal**

Previous research had 3 main directions. Either, the hourly prediction of the occurrence of a sickness, death or 
complications (like sepsis) is intended. The goal here is to offer earlier prediction than conventional scoring-methods.
For this, hourly time-series data is used as a basis. This is rather complex and many times, supportive features 
need to be derived first.

A second direction is the prediction of the occurrence of a sickness based on general data (averages, not hourly).
For this, a precise knowledge of the medical factors is required.

The last case is to predict the general development of a patient. Different approaches are possible: 
length of stay, probability of a secondary stay (relapse) and mortality prediction.
This last case is the intention of this work: 
the prediction of death within the hospital stay, within 30 days and within 360 days.

**Output:**

An overview of all selected patients is created in '0_patient_cohort.csv'. For each patient a .csv file is created
with their hourly chart-events (time-series data). These time-series data will be converted to averages, min and max 
features in a later step.

**Estimated Time:** 

Getting access to the MIMIC-III dataset by PhysioNet takes between 1-4 weeks. 
For this, also a course regarding data-protection had to be conducted, which regularly takes about 4 hours.
The download and import of the data into the postgre database takes about 5 hours.

Running the set-up for the table 'all_diagnoses_icd' takes about 45 minutes.
There, for each admission all available diagnoses are saved in the new field 'all_icd_codes' as an array. 
Thus, in later filtering for only one diagnoses-type, all other diagnoses for this admission are not lost.

Creation of the patient_cohort can be calculated very quickly (10 seconds). 

However, the creation of the single patient files with all their chart-events takes up multiple hours. 
Approximately the creation of one patient .csv file takes 30 seconds. 
Depending on the use case, for example stroke there are 1400 available patients, which leads to about 700 minutes (>10h).

**Filtering:** 

There are 3 steps where the dataset is filtered.
1) Patient Selection (not changeable):

The first filtering determines, which patient is suitable. While there are 58.976 unique admissions in the dataset, 
many patients are underage or need to be excluded because there is missing data.
Also, only patients, for which the 'metavision' system was used, are included. This is regretful, as 'CareVue' patients make up
a large amount. However, the different software that was used led to different chart_event-ids /item_ids. This makes it 
not suitable to use both systems. Metavision was chosen, as it is the newer one and most other researchers use it.
This also means, that instead of around 12.000  (often duplicate) aspects that were measured on the ICU, only 2.200 remain.

This filtering was in parts based on concepts from previous research by Alistair et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5851804/pdf/nihms925667.pdf).
It should not be changed by the user. After this first step, there are still 16.058 unique icu-stays left. 

2) Use-Case Selection (Input at the 'export_patients_to_csv()' function):
 
The patients can be filtered for their illnesses with the icd9-codes (diagnoses at admission). 
A user can change the selected icd9-codes in the supplement file 'selection_icd9_codes.py'.
This step is fundamental and probably needs some medical background knowledge. 

Moreover, some icd9_titles might seem like they have medical relevance, but they were used differently in the actual
diagnosis of patients for MIMIC-III.
For example, not all cerebrovascular icd9_code titles that sound like they indicate stroke can actually be accounted as stroke cases.
The code '43883 - Facial Weakness' might be considered as a valid indicator. However, after cross-referencing 
the 'diagnosis_text' for all the patients with 43883, it became clear that those were mainly heart-attack patients. 
Thus, 43883 was removed as a selector for stroke. Of course, if one of those removed patients also has another actually
valid icd9_code, like '430 - Subarachnoid hemorrhage', then they were kept in the dataset. This example was meant to 
explain, that it is crucial to select the correct icd9_codes, as the right patient selection is one of the most important steps of the analysis.

Another helpful file to inspect all available icd9-codes and map codes to their titles is 'icd9_codes_dictionary.csv'.
This file can also be created with the SQL script 'create_icd9_codes_dictionary.sql'. 
Important: Many procedures have the same code as diagnoses. They must be considered separately.

There are 1.451 unique icustay_ids for the use-case of stroke (about 9.5% of the available data). 
A patient can come to the ICU multiple times, within one hospital stay (one admission). 
Thus, the relevant key shall be the icu-stay, to not use duplicate patients.

3) Feature Selection (Input at the 'export_patients_to_csv()' function):

The relevant labels/features can be chosen by the user. The complete dataset offers 12.487 itemids (for any kind of chart_event that happened at the ICU). These are too many features for any useful analysis. But as stated in 1), only metavision patients are included. Metavision enables 2.992 itemids. This would still be too many features.
Thus, a user has to choose, which labels (itemids) will be important for the respective use-case. 
This can be selected in the supplement file 'selection_itemids_list_mv.py'.

A helpful file to inspect and map itemids to their labels and unit-of-measurements is the file 'events_dictionary.csv'.
This file can also be created with the SQL script 'create_events_dictionary.sql'.

It is recommended to choose about 40 itemids. This selection naturally requires some medical knowledge, 
thus it is also recommended to use previous research filtering as a guideline. 
Patient demographics, secondary diagnosis will always be derived, regardless of this filter. 


4) Notes for Future Research:

There are many labels in the MIMIC-III dataset that have not yet been included into this analysis.
It is possible to include sources like: transfers, services, microbiology-events, cptevents, prescriptions for further research.
For example, the amount of transfers or a special kind of prescriptions might be strong indicators for relapse-rates. 
However, these features are related to the treatment and behavior of the medical staff. They are not directly related
to the illness itself and thus, where not included in this analysis.
Especially the table note-events offers a noteworthy source for possible ML-research based on text-analysis.

All of these sources might be included inside the SQL-function 'get_all_events_view' for future research. 

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