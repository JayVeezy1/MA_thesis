# Master Thesis by Jakob Vanek

<p align="center">
  <img src="supplements/title_page.png">
</p>

**Analysis Goal:** 
The goal of this thesis is to examine the potential of mortality prediction within the MIMIC-III dataset.
The selected use case is 'stroke' with 2,655 patients.
The final thesis paper, with the analysis results, can be provided upon request. 

# Part 0 - Setup
**MIMIC-III Setup in Postgres**

The initial part of this project prepares the Medical Information Mart for Intensive Care dataset (MIMIC-III, v1.4) dataset for analysis. 
The data must be imported and set up in the Postgres database (instructions available at https://physionet.org/content/mimiciii/1.4/).
Also, in the 'supplements/SQL' folder, necessary SQL scripts are provided, to set up further required PL/pgSQL functions. 
Once this setup is finished a user can run all functions from the main.py Python script.

These Python scripts can be used to export patients (based on icustay_ids) from the local Postgres database into individual .csv files.

The actual data for the MIMIC-III dataset can not be shared here. It is available, upon request, through PhysioNet.

**Patient Export**

For each patient, a .csv file is exported from the local Postgres DB with their hourly chart events (time-series data). These time-series data will be converted to averages in a later step. 
The resulting 'average-patient-cohort.csv' file will be the foundation for all subsequent analysis steps. 
This is also the file, that a user can upload in the supplement frontend to visualize the analysis.

**Estimated Time:** 

Getting access to the MIMIC-III dataset by PhysioNet takes between 1-4 weeks. 
For this, also a course regarding data protection has to be conducted, which regularly takes about 4 hours.
The download and import of the data into the Postgres database takes about 5 hours.

Running the set-up for the table 'all_diagnoses_icd' takes about 45 minutes.
There, for each admission, all available diagnoses are saved in the new field 'all_icd_codes' as an array. 
Thus, in later filtering for only one diagnoses type, all other diagnoses for this admission are not lost.

Creation of the patient_cohort can be calculated very quickly (10 seconds). 

However, the creation of the single patient files with all their chart-events takes up multiple hours. 
Approximately the creation of one patient .csv file takes 30 seconds. 
Depending on the use case, for example, stroke there are 2600 available patients, which leads to about 1300 minutes (~20h).
It is recommended to run this export of patients from the Postgres DB into individual .csv files overnight.

**Requirements:**

A Python Development Environment is needed and the required packages are listed in requirements.txt.
Moreover, it is recommended to not use a Python Version above 3.10.9 as PacMap is not yet compatible with any newer version.
Currently, there is no alternative available, but it might be possible in future PacMap versions, to also use the latest Python Version.
There are no additional requirements for the deep learning models. However, in the future GPU-based calculations might be added, which would require certain hardware.

**Filtering:** 

There are 3 steps where the dataset is filtered.
1) Patient Selection (not changeable SQL files):

The first filtering determines, which patient is suitable. While there are 58.976 unique admissions in the dataset, 
many patients are underage or need to be excluded because there is missing data.
Currently, patients from the 'metavision' as well as the 'CareVue' dataset are included. 

This filtering was in parts based on concepts from previous research by Alistair et al. (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5851804/pdf/nihms925667.pdf).
It should not be changed by the user. After this first step, there are still 13.383 unique icu-stays left. 

2) Use-Case Selection (Input at the 'export_patients_to_csv()' function):
 
The patients can be filtered for their illnesses with the icd9-codes (diagnoses at admission). 
A user can change the selected icd9-codes in the supplement file 'selection_icd9_codes.py'.
This step is fundamental and probably needs some medical background knowledge. 

Moreover, some icd9_titles might seem like they have medical relevance, but they were used differently in the actual
diagnosis of patients for MIMIC-III.
For example, not all cerebrovascular icd9_code titles that sound like they indicate stroke can actually be accounted as stroke cases.
The code '43883 - Facial Weakness' might be considered a valid indicator. However, after cross-referencing 
the 'diagnosis_text' for all the patients with 43883, it became clear that those were mainly heart-attack patients. 
Thus, 43883 was removed as a selector for stroke. Of course, if one of those removed patients also has another actually
valid icd9_code, like '430 - Subarachnoid hemorrhage', then they were kept in the dataset. This example was meant to 
explain, that it is crucial to select the correct icd9_codes, as the right patient selection is one of the most important steps of the analysis.

Another helpful file to inspect all available icd9-codes and map codes to their titles is 'icd9_codes_dictionary.csv'.
This file can also be created with the SQL script 'create_icd9_codes_dictionary.sql'. 
Important: Many procedures have the same code as diagnoses. They must be considered separately.

The final selection of icd9-codes for stroke was chosen based on the stroke-type definition from https://health.mo.gov/data/mica/CDP_MICA/StrokeDefofInd.html :

- Hemorrhage: 430, 431, 432, 4329

- Ischemic (+TIA): 433, 4330, 4331, 4332, 434, 4340, 43400, 43401, 4341, 43411, 435, 4350, 4351, 4353, 4359, 436

- Other (+late_effects_of_stroke): 437, 4370, 4371, 4372, 4373, 4374, 438, 4381, 43811, 4382, 43820, 4383, 4384, 4385, 4388, 43882, 43885

Also, all icu-stays with less than 24 hours were removed, which resulted in a total of 2,655 icu-stays  (about 9.5% of the available data).
A patient can come to the ICU multiple times, within one hospital stay (one admission). 
Thus, the relevant key shall be the ICU stay, to not use duplicate patients.

**Further Info:**
When checking the .csv files manually, be sure to turn off excels column-data-transformation. 
Otherwise, some comma numbers, which are separated with a '.' will appear as dates and then be transformed to very large numbers.

3) Feature Pre-Selection (Input at the 'export_patients_to_csv()' function):

All available features for each patient will are exported and then inside Python the features with the highest relevance
(depending on previous research) are selected.

The relevant labels/features can be chosen by the user. The carevue database offers 12.487 itemids (for any kind of chart_event that happened at the ICU). These are too many features for any useful analysis. Metavision enables 2.992 itemids. This would still be too many features.
Thus, a user has to choose, which labels (itemids) will be important for the respective use case. 
This can be selected in the supplement file 'selection_itemids_list.py'.

A helpful file to inspect and map itemids to their labels and unit-of-measurements is the file 'events_dictionary.csv'.
This file can also be created with the SQL script 'create_events_dictionary.sql'.

It is recommended to choose about 40 itemids. This selection naturally requires some medical knowledge, 
thus it is also recommended to use previous research filtering as a guideline. 
Patient demographics, secondary diagnosis will always be derived, regardless of this filter.

Features that are always included:
- Patient Vitals -> need to be selected after analysis of relevance
- OASIS Score and related features (mechanical ventilation, GCS Score)
- General Patient Information (subject_id, hadm_id, icustay_id, demographics)

The actual selection out of these 40 features can be changed for each analysis step within the subsequent processes and within the frontend.

**Additional Information about the OASIS Score Feature**

The OASIS score represents a very common and important index for patient risk. It is commonly used to compare
new prediction models with the status quo. The calculation of the score depends on a multitude of factors.
As these steps have already been implemented by previous research and made available via GitHub, their code for the
SQL scripts were used for this analysis (source: https://github.com/caisr-hh/Dayly-SAPS-III-and-OASIS-scores-for-MIMIC-III).
The used scripts can be found in supplements/SQL/Dayly-SAPS-III-and-OASIS-scores-for-MIMIC-III-master 
They must be loaded manually into the postgres database before running the patient selection. The time needed to execute
the scripts is about 15 minutes in total (each ranging between 30 seconds and 3 minutes). 
The created views are saved inside the MIMIC-III schema, and not inside the public schema, where all the other objects 
of this thesis were created. This makes it easier to see, which scripts come from where. 

# Actual Analysis

The implementation of the following steps makes up the core of my master thesis.

## Part 1 - CSV Import & Preprocessing
    # Step 1.1) Import all .csv files as an individual 'Patient' Object with a related dataframe
    # NOTE: the variable PROJECT_PATH has to be adjusted to a user's local system

    # Step 1.2) Calculate avg, min, max for each feature for each patient
    # Depending on missing value-rate impute data
    # Also factorize and scale data

    # Step 1.3) Combine all selected patients into a 'set' object, and save this object as .pickle to cache it
    # Based on this cached set create the main file 'average_patient_cohort.csv', this is used for all further analysis
    
## Part 2 - Data Analysis
    # Step 2.1) General Statistics
    # includes functions to calculate General Overview Table and Deaths Overview Table
    
    # Step 2.2) Correlations
    
    # Step 2.3) Clustering with PacMap Visualization
    # including kMeans, kPrototypes, DBSCAN, SLINK

## Part 3 - Machine Learning Predictions
    # Step 3.1) Random Forest
    # including GradientBoost

    # Step 3.2) XGBoost

    # Step 3.3) Sequential Deep Learning/Neural Network

    # Step 3.4) Feature Importance based on SHAP


## Part 4 - Fairness Metrics
    # Step 4.1) Calculate Fairness across manually selected Subgroups (select a feature like 'Gender' and a privileged class/value)


## Part 5 - Subgroup Analysis
    # Step 5.1) Detect Subgroups based on clustering and entropy

    # Step 5.2) Calculate fairness and feature importance between selected Cluster/Subgroup (=privileged class) and the rest of data

## Part 6 - Frontend
    # A streamlit frontend was developed for this thesis to visualize each of the previous chapters
    # It can be started directly by running the 'main.py' file
    # Herein, a user has to upload the 'average_patient_cohort.csv' file. 
    
In conclusion, a user must have access to the MIMIC-III dataset and at least implement the setup (Part 0) and pre-processing (Part 1) to visualize the results of this thesis in the frontend. 

Overall, the code for this thesis offers a broad foundation for research within the MIMIC-III dataset. If any questions arise, please, feel free to get in contact.
