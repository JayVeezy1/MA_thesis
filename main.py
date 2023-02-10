import pandas as pd

from objects.patients import Patient
from step_1_setup_data import cache_IO, mimic_to_csv, select_relevant_features
from step_3_data_analysis import correlations, classification, clustering, general_statistics, data_visualization
from supplements import selection_icd9_codes

####### MAIN #######
if __name__ == '__main__':
    PROJECT_PATH: str = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'  # this variable must be fitted to the users local project folder
    USE_CASE_NAME: str = 'stroke_all_systems'  # stroke_patients_data       # heart_infarct_patients_data

    FEATURES_DF = pd.read_excel('./supplements/feature_preprocessing_table.xlsx')
    SELECTED_FEATURES = FEATURES_DF[FEATURES_DF['selected_for_analysis'] == 'yes']['feature_name'].to_list()
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
    cache_IO.load_data_from_cache(project_path=PROJECT_PATH, features_df=FEATURES_DF,
                                  use_case_name=USE_CASE_NAME, delete_existing_cache=False)

    ### Preprocessing
    # Step 2) Calculate Avg, Normalize, Impute & Interpolate for each patient
    complete_avg_patient_cohort = Patient.get_avg_patient_cohort(project_path=PROJECT_PATH, use_case_name=USE_CASE_NAME,
                                                                 selected_patients=[])  # if empty -> all

    # avg_hemorrhage_cohort = complete_avg_patient_cohort[complete_avg_patient_cohort['stroke_type'] == 'hemorrhagic']
    # avg_ischemic_cohort = complete_avg_patient_cohort[complete_avg_patient_cohort['stroke_type'] == 'ischemic']

    # TODO: add normalization

    # todo long term: add interpolation/imputation to timeseries depending on NaN, and also min/max-columns for features?

    ### Data Analysis
    # Step 3.1) General Statistics
    general_statistics.calculate_deaths_table(selected_patient_cohort=complete_avg_patient_cohort,
                                              cohort_title='complete_avg_patient_cohort',
                                              use_case_name=USE_CASE_NAME,
                                              selected_features=SELECTED_FEATURES,
                                              save_to_file=True)
    # label | classification (existing values of this feature) | patients_in_training_set (count/occurrence) | correlation_to_death | p-value (continuous) | chi-squared-value (categorical) | NaN Amount

    # TODO: Rework filtering because its not correct. -> maybe move stroke_type and infarct_type into Patient Class -> more flexible for future cases instead of Postgres Script
    # todo: are the hemorrhagic filters correct?
    # TODO: IMPORTANT: Do calculations again with correct dataset (for stroke & heart new exports) and with normalized data??? -> better results??
    general_statistics.calculate_feature_overview_table(selected_patient_cohort=complete_avg_patient_cohort,
                                                        features_df=FEATURES_DF,
                                                        cohort_title='complete_avg_patient_cohort',
                                                        use_case_name=USE_CASE_NAME,
                                                        selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                        selected_features=SELECTED_FEATURES,
                                                        save_to_file=True)

    # Step 3.2) Visualization, Correlation, Clustering, etc.
    # PacMap
    # data_visualization.display_pacmap(avg_patient_cohort=avg_hemorrhage_cohort, cohort_title='avg_hemorrhage_cohort',
    #                                    use_case_name = USE_CASE_NAME,
    #                                   selected_features=SELECTED_FEATURES, selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                                   save_to_file=True)

    # Correlations (also available: plot_heatmap and plot_pairplot)
    # correlations.plot_correlations(avg_patient_cohort=avg_hemorrhage_cohort,
    #                                    use_case_name = USE_CASE_NAME,
    #                              cohort_title='avg_hemorrhage_cohort',
    #                             selected_features=SELECTED_FEATURES,
    #                            selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE, save_to_file=False)

    # todo long term: Test DBSCAN Prototype, implement both available methods in one main-clustering method 'plot_clusters_on_pacmap'
    # Clustering Prototype
    # clustering.plot_sh_score_kmeans(avg_patient_cohort=avg_hemorrhage_cohort,
    #                                    use_case_name = USE_CASE_NAME, cohort_title='avg_hemorrhage_cohort', selected_features=SELECTED_FEATURES, selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE, filter_labels=False, save_to_file=True)
    # manually checking silhouette score shows: 4 clusters is optimal
    # SELECTED_CLUSTERS_COUNT = 4
    # clustering.plot_clusters_on_pacmap(avg_patient_cohort=avg_hemorrhage_cohort, cohort_title='avg_hemorrhage_cohort',
    #                                    use_case_name = USE_CASE_NAME,
    #                                   selected_features=SELECTED_FEATURES, selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                                 selected_cluster_count=SELECTED_CLUSTERS_COUNT, filter_labels=False,
    #                                save_to_file=False)

    # Analysing Single Cluster Prototype        # todo check: why is cluster 1 in above example only 1 patient? -> icustay_id = 293675, but the patient actually has most data available. Where is this error coming from? Or is it an indicator that the cluster and icustay_id is not correctly mapped?
    # cluster_to_analyse: int = 2
    # #filtered_cluster_icustay_ids: list = clustering.get_ids_for_cluster(avg_patient_cohort=avg_hemorrhage_cohort,
    #                                                                 cohort_title='avg_hemorrhage_cohort',
    #                                    use_case_name = USE_CASE_NAME,

    #                                                                selected_features=SELECTED_FEATURES,
    #                                                               selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                                                              selected_k_means_count=SELECTED_CLUSTERS_COUNT,
    #                                                             selected_cluster=cluster_to_analyse,
    #                                                            filter_labels=False)
    # filtered_cluster_cohort = complete_avg_patient_cohort[
    #    complete_avg_patient_cohort['icustay_id'].isin(filtered_cluster_icustay_ids)]
    # general_statistics.calculate_feature_overview_table(selected_patient_cohort=filtered_cluster_cohort,
    #                                    use_case_name = USE_CASE_NAME,

    #                                                   cohort_title='filtered_cluster_cohort',
    #                                                  selected_features=SELECTED_FEATURES, save_to_file=False)

    ### Machine Learning Predictions
    # Step 4.1) Random Forest
    # classification.calculate_RF_on_cohort(avg_patient_cohort=avg_hemorrhage_cohort,
    #                                    cohort_title='avg_hemorrhage_cohort',
    #                                    use_case_name = USE_CASE_NAME,

    #                                   selected_features=SELECTED_FEATURES,
    #                                  selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                                 save_to_file=True)

# Step 4.2) XGBoost

# Step 4.3) Deep Learning/Neural Network


### Fairness Metrics
# Step 5.1) Calculate Fairness for manual Subgroups


### Automated Subgroup detection
# Step 6.1) Calculate automated Subgroups and related fairness metrics

# Step 6.2) Include ASDF-Dashboard as frontend
