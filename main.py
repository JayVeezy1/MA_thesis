import pandas as pd

from objects.patients import Patient
from step_1_setup_data import cache_IO, mimic_to_csv, select_relevant_features
from step_2_preprocessing.preprocessing_functions import get_preprocessed_avg_cohort_and_features
from step_3_data_analysis import correlations, classification, clustering, general_statistics, data_visualization
from supplements import selection_icd9_codes

####### MAIN #######
if __name__ == '__main__':
    PROJECT_PATH: str = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'  # this variable must be fitted to the users local project folder
    USE_CASE_NAME: str = 'stroke_all_systems'  # stroke_patients_data       # heart_infarct_patients_data
    FEATURES_DF = pd.read_excel('./supplements/feature_preprocessing_table.xlsx')
    SELECTED_DEPENDENT_VARIABLE = 'death_in_hosp'  # death_3_days, death_in_hosp

    ### Setup, MIMIC-III Export from DB, Load from Cache
    # Step 0) Setup when first time using db:
    # mimic_to_csv.setup_postgre_files()                 # setup all needed background functions and views for postgre. Warning: Sometimes this setup from Python does not work. Then you simply copy&paste each SQL Script into PostGre QueryTool and execute it.
    # mimic_to_csv.create_table_all_diagnoses()          # create a necessary table 'all_diagnoses' where for each admission all available diagnoses are saved in the new field 'all_icd_codes' (takes approx. 2 hours)
    # mimic_to_csv.create_supplement_dictionaries()      # create supplementary dictionary files
    # mimic_to_csv.load_comorbidities_into_db()          # create the necessary table 'comorbidity_codes' where the icd9_codes that are used to find important comorbidities are loaded into the DB

    # Step 1.1) Export the raw patient data for the specified use_case (icd_list) into .csv files, all available features will be exported
    # metavision stroke use-case has 1232 patients, each takes approx. 30 seconds -> 500 Minutes, 8,5 hours
    # complete stroke cases has 2655 -> 1300 minutes, 20 hours
    # Run this function only once for the patient-export. Afterwards use .csvs
    # mimic_to_csv.export_patients_to_csv(project_path=PROJECT_PATH,
    #                                   use_case_icd_list=selection_icd9_codes.selected_stroke_codes,            # stroke case = icd9_00_stroke_selected
    #                                  use_case_itemids=[],
    #                                 use_case_name=USE_CASE_NAME)

    # Step 1.2) Filter final patient.csvs for relevant features and export as 'final_dataset'
    # transform raw.csvs into filtered, final .csvs, also transform carevue feature-names into metavision names (not ideal)
    # select_relevant_features.export_final_dataset(project_path=PROJECT_PATH, use_case_name=USE_CASE_NAME)

    # Step 1.3) Load all .csv files as a 'Patient' Object, use Pickle for Cache
    cache_IO.load_data_from_cache(project_path=PROJECT_PATH, features_df=FEATURES_DF, use_case_name=USE_CASE_NAME,
                                  delete_existing_cache=False)

    ### Preprocessing
    # Step 2) Calculate Avg, Filter, Scale, Impute & Interpolate for each patient
    # Choose: dbsource filter
    complete_avg_cohort = Patient.get_avg_patient_cohort(PROJECT_PATH, USE_CASE_NAME, selected_patients=[])  # empty=all
    metavision_avg_cohort = complete_avg_cohort[complete_avg_cohort['dbsource'] == 'metavision']
    carevue_avg_cohort = complete_avg_cohort[complete_avg_cohort['dbsource'] == 'carevue']
    # Choose: stroke_type filter, also option: change complete_avg_cohort to metavision_avg_cohort or carevue_avg_cohort
    hemorrhage_avg_cohort = complete_avg_cohort[complete_avg_cohort['stroke_type'] == 'hemorrhagic']
    ischemic_avg_cohort = complete_avg_cohort[complete_avg_cohort['stroke_type'] == 'ischemic']
    # Choose: scaled_cohort
    scaled_complete_avg_cohort = Patient.get_avg_scaled_data(complete_avg_cohort.copy())
    scaled_hemorrhage_avg_cohort = Patient.get_avg_scaled_data(hemorrhage_avg_cohort.copy())
    scaled_ischemic_avg_cohort = Patient.get_avg_scaled_data(ischemic_avg_cohort.copy())
    # Choose: Cohort Parameters
    SELECTED_COHORT = scaled_complete_avg_cohort
    SELECTED_COHORT_TITLE = 'scaled_complete_avg_cohort'
    SELECT_SAVE_FILES = False
    # Automated: Preprocessed Cohort
    SELECTED_COHORT_preprocessed, SELECTED_FEATURES = get_preprocessed_avg_cohort_and_features(
        avg_cohort=SELECTED_COHORT,
        cohort_title=SELECTED_COHORT_TITLE,
        features_df=FEATURES_DF)

    # TODO NEXT STEP: XGBoost + normalized data + hemorrhagic + filtered_features -> better results? Otherwise change to heart?

    # todo long term: add interpolation/imputation/outliers to timeseries depending on NaN
    # todo if keeping stroke: are the hemorrhagic filters correct? -> maybe move filtering of stroke_type and infarct_type into Patient Class -> more flexible for future cases instead of Postgres Script

    ### Data Analysis
    # Step 3.1) General Statistics
    # general_statistics.calculate_deaths_table(selected_patient_cohort=SELECTED_COHORT_preprocessed,
    #                                        cohort_title=SELECTED_COHORT_TITLE,
    #                                       use_case_name=USE_CASE_NAME,
    #                                      save_to_file=True)

    # general_statistics.calculate_feature_overview_table(selected_patient_cohort=SELECTED_COHORT_preprocessed,
    #                                                   features_df=FEATURES_DF,
    #                                                  selected_features=SELECTED_FEATURES,
    #                                                 cohort_title=SELECTED_COHORT_TITLE,
    #                                                use_case_name=USE_CASE_NAME,
    #                                               selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                                              save_to_file=True)

    # Step 3.2) Correlation
    # Correlations (also available: plot_heatmap and plot_pairplot)
    # correlations.plot_correlations(avg_patient_cohort=SELECTED_COHORT_preprocessed,
    #                              use_case_name=USE_CASE_NAME,
    #                             cohort_title=SELECTED_COHORT_TITLE,
    #                            features_df=FEATURES_DF,
    #                           selected_features=SELECTED_FEATURES,
    #                          selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                         save_to_file=SELECT_SAVE_FILES)

    # Step 3.3) Visualization (PacMap)
    # data_visualization.display_pacmap(avg_patient_cohort=SELECTED_COHORT_preprocessed,
    #                                 use_case_name=USE_CASE_NAME,
    #                                cohort_title=SELECTED_COHORT_TITLE,
    #                               features_df=FEATURES_DF,
    #                              selected_features=SELECTED_FEATURES,
    #                             selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                            save_to_file=SELECT_SAVE_FILES)

    # Step 3.4) Clustering (kmeans, DBSCAN, ...)
    # kmeans
    # clustering.plot_sh_score_kmeans(avg_patient_cohort=SELECTED_COHORT_preprocessed,
    #                               cohort_title=SELECTED_COHORT_TITLE,
    #                              use_case_name=USE_CASE_NAME,
    #                             features_df=FEATURES_DF,
    #                            selected_features=SELECTED_FEATURES,
    #                           selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                          save_to_file=SELECT_SAVE_FILES)
    SELECTED_CLUSTERS_COUNT = 4  # manually checking silhouette score shows optimal cluster count (higher is better)
    # clustering.plot_k_means_on_pacmap(avg_patient_cohort=SELECTED_COHORT_preprocessed,
    #                                 cohort_title=SELECTED_COHORT_TITLE,
    #                                use_case_name=USE_CASE_NAME,
    #                               features_df=FEATURES_DF,
    #                              selected_features=SELECTED_FEATURES,
    #                             selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                            selected_cluster_count=SELECTED_CLUSTERS_COUNT,
    #                           save_to_file=SELECT_SAVE_FILES)

    # DBSCAN
    # todo long term: plot sh_score_DBSCAN
    # todo long term: Test DBSCAN Prototype, implement both available methods in one main-clustering method 'plot_clusters_on_pacmap'

    # Cluster Comparison
    # clusters_overview_table = clustering.calculate_clusters_overview_table(selected_cohort=SELECTED_COHORT_preprocessed,
    #                                                                      cohort_title=SELECTED_COHORT_TITLE,
    #                                                                     use_case_name=USE_CASE_NAME,
    #                                                                    selected_clusters_count=SELECTED_CLUSTERS_COUNT,
    #                                                                   features_df=FEATURES_DF,
    #                                                                  selected_features=SELECTED_FEATURES,
    #                                                                 selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                                                                save_to_file=SELECT_SAVE_FILES)

    ### Machine Learning Predictions
    # Step 4.1) Classification: RandomForest & XGBoost
    SELECTED_CLASSIFICATION_METHOD = 'RandomForest'  # options: RandomForest | XGBoost
    SELECTED_SAMPLING_METHOD = 'no_sampling'         # options: no_sampling | oversampling | undersampling
    classification.calculate_classification_on_cohort(classification_method=SELECTED_CLASSIFICATION_METHOD,
                                                      sampling_method=SELECTED_SAMPLING_METHOD,
                                                      avg_cohort=SELECTED_COHORT_preprocessed,
                                                      cohort_title=SELECTED_COHORT_TITLE,
                                                      use_case_name=USE_CASE_NAME,
                                                      features_df=FEATURES_DF,
                                                      selected_features=SELECTED_FEATURES,
                                                      selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                      save_to_file=SELECT_SAVE_FILES
                                                      )

# todo after XGBoost: classification comparison table: list[all_cohorts], list[prediction_types], list[dependent_variables] -> for each ... for each ... -> one final overview table to compare all accuracy & recall results
# Step 4.3) Deep Learning/Neural Network


### Fairness Metrics
# Step 5.1) Calculate Fairness for manual Subgroups


### Automated Subgroup detection
# Step 6.1) Calculate automated Subgroups and related fairness metrics

# Step 6.2) Include ASDF-Dashboard as frontend
