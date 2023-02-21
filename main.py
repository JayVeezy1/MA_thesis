import pandas as pd

from objects.patients import Patient
from step_1_setup_data import cache_IO, mimic_to_csv, select_relevant_features
from step_2_preprocessing.preprocessing_functions import get_preprocessed_avg_cohort
from step_3_data_analysis import correlations, classification, clustering, general_statistics, data_visualization
from supplements import selection_icd9_codes

####### MAIN #######
# By: Jakob Vanek, 2023
if __name__ == '__main__':
    PROJECT_PATH: str = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'  # this variable must be fitted to the users local project folder
    USE_CASE_NAME: str = 'stroke_all_systems'  # stroke_patients_data       # heart_infarct_patients_data
    FEATURES_DF = pd.read_excel('./supplements/feature_preprocessing_table.xlsx')
    SELECTED_DEPENDENT_VARIABLE = 'death_in_hosp'  # death_3_days, death_in_hosp
    ALL_DEPENDENT_VARIABLES: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days',
                                     'death_365_days']  # FEATURES_DF.loc[FEATURES_DF['potential_for_analysis'] == 'prediction_variable', 'feature_name'].to_list()

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
    # Options: dbsource filter
    complete_avg_cohort = Patient.get_avg_patient_cohort(PROJECT_PATH, USE_CASE_NAME, selected_patients=[])  # empty=all
    metavision_avg_cohort = complete_avg_cohort[complete_avg_cohort['dbsource'] == 'metavision']
    carevue_avg_cohort = complete_avg_cohort[complete_avg_cohort['dbsource'] == 'carevue']
    # Options: stroke_type filter, also option: change complete_avg_cohort to metavision_avg_cohort or carevue_avg_cohort
    hemorrhage_avg_cohort = complete_avg_cohort[complete_avg_cohort['stroke_type'] == 'hemorrhagic']
    ischemic_avg_cohort = complete_avg_cohort[complete_avg_cohort['stroke_type'] == 'ischemic']
    # Options: scaled_cohort (recommended)
    scaled_complete_avg_cohort = Patient.get_avg_scaled_data(complete_avg_cohort.copy())
    scaled_hemorrhage_avg_cohort = Patient.get_avg_scaled_data(hemorrhage_avg_cohort.copy())
    scaled_ischemic_avg_cohort = Patient.get_avg_scaled_data(ischemic_avg_cohort.copy())

    # CHOOSE: Cohort Parameters
    SELECTED_COHORT = scaled_complete_avg_cohort
    SELECTED_COHORT_TITLE = 'scaled_complete_avg_cohort'
    SELECT_SAVE_FILES = True
    # Automated: Preprocessed Cohort
    SELECTED_COHORT_preprocessed = get_preprocessed_avg_cohort(avg_cohort=SELECTED_COHORT,
                                                               cohort_title=SELECTED_COHORT_TITLE,
                                                               features_df=FEATURES_DF)
    SELECTED_FEATURES = list(SELECTED_COHORT_preprocessed.columns)

    # Automated: List of all cohorts_preprocessed for model comparison
    scaled_complete_cohort_preprocessed = get_preprocessed_avg_cohort(avg_cohort=scaled_complete_avg_cohort,
                                                                      cohort_title='scaled_complete_avg_cohort',
                                                                      features_df=FEATURES_DF)
    scaled_hemorrhage_cohort_preprocessed = get_preprocessed_avg_cohort(avg_cohort=scaled_hemorrhage_avg_cohort,
                                                                        cohort_title='scaled_hemorrhage_avg_cohort',
                                                                        features_df=FEATURES_DF)
    scaled_ischemic_cohort_preprocessed = get_preprocessed_avg_cohort(avg_cohort=scaled_ischemic_avg_cohort,
                                                                      cohort_title='scaled_ischemic_avg_cohort',
                                                                      features_df=FEATURES_DF)
    ALL_COHORTS_WITH_TITLES: dict = {'scaled_complete_avg_cohort': scaled_complete_cohort_preprocessed,
                                     'scaled_hemorrhage_avg_cohort': scaled_hemorrhage_cohort_preprocessed,
                                     'scaled_ischemic_avg_cohort': scaled_ischemic_cohort_preprocessed}
    print('STATUS: Preprocessing finished. \n')

    # todo check: are the hemorrhagic filters for stroke actually correct? Can I use carevue cases or only metavision?
    # todo check: why is curve of sh_score so bad

    # TODO today: Finish clustering_overview_function
    # possible to automatically find optimal sh_score? -> why is my sh_score so bad? -> too many features?

    # TODO this week: correlation of flag-features must be calculated differently than continuous! -> How? -> then add these into predictions as well!
        # Rho Test
        # Chi-Squared Test
        # add these to the feature+correlation overview table


    # todo next week: add a Deep Learning model
    # todo next week write: which model has best results for now? -> this is a first part-result, should be comparable to papers
    # todo after: analyze clusters for 'fairness' -> bridge to ASDF Dashboard
    # todo long term: add 'decision-boundary-plot' to visualize the clustering (on 2 features)

    ### Data Analysis
    # Step 3.1) General Statistics
    general_statistics.calculate_deaths_table(use_this_function=False,  # True | False
                                              selected_cohort=SELECTED_COHORT_preprocessed,
                                              cohort_title=SELECTED_COHORT_TITLE,
                                              use_case_name=USE_CASE_NAME,
                                              save_to_file=True)

    general_statistics.calculate_feature_overview_table(use_this_function=False,  # True | False
                                                        selected_cohort=SELECTED_COHORT_preprocessed,
                                                        features_df=FEATURES_DF,
                                                        selected_features=SELECTED_FEATURES,
                                                        cohort_title=SELECTED_COHORT_TITLE,
                                                        use_case_name=USE_CASE_NAME,
                                                        selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                        save_to_file=True)

    # Step 3.2) Correlation
    # Correlations
    correlations.plot_correlations(use_this_function=False,  # True | False
                                   use_plot_heatmap=False,
                                   use_plot_pairplot=False,
                                   cohort_title=SELECTED_COHORT_TITLE,
                                   selected_cohort=SELECTED_COHORT_preprocessed,
                                   features_df=FEATURES_DF,
                                   selected_features=SELECTED_FEATURES,
                                   selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                   use_case_name=USE_CASE_NAME,
                                   save_to_file=SELECT_SAVE_FILES)

    # Step 3.3) Visualization (PacMap)
    data_visualization.display_pacmap(use_this_function=False,  # True | False
                                      selected_cohort=SELECTED_COHORT_preprocessed,
                                      use_case_name=USE_CASE_NAME,
                                      cohort_title=SELECTED_COHORT_TITLE,
                                      features_df=FEATURES_DF,
                                      selected_features=SELECTED_FEATURES,
                                      selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                      save_to_file=SELECT_SAVE_FILES)

    # Step 3.4) Clustering (kmeans, DBSCAN, ...)
    # kmeans: sh_score
    clustering.plot_sh_score_kmeans(use_this_function=False,  # True | False
                                    selected_cohort=SELECTED_COHORT_preprocessed,
                                    cohort_title=SELECTED_COHORT_TITLE,
                                    use_case_name=USE_CASE_NAME,
                                    features_df=FEATURES_DF,
                                    selected_features=SELECTED_FEATURES,
                                    selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                    save_to_file=SELECT_SAVE_FILES)
    # kmeans: plotting
    SELECTED_KMEANS_CLUSTERS_COUNT = 9  # manually checking silhouette score shows optimal cluster count (higher is better)
    clustering.plot_k_means_on_pacmap(use_this_function=False,  # True | False
                                      selected_cohort=SELECTED_COHORT_preprocessed,
                                      cohort_title=SELECTED_COHORT_TITLE,
                                      use_case_name=USE_CASE_NAME,
                                      features_df=FEATURES_DF,
                                      selected_features=SELECTED_FEATURES,
                                      selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                      selected_cluster_count=SELECTED_KMEANS_CLUSTERS_COUNT,
                                      save_to_file=SELECT_SAVE_FILES)

    # kmeans: Cluster Comparison (only for manually selected_cluster_count -> only kmeans)
    clusters_overview_table = clustering.calculate_clusters_overview_table(use_this_function=True,  # True | False
                                                                           selected_cohort=SELECTED_COHORT_preprocessed,
                                                                           cohort_title=SELECTED_COHORT_TITLE,
                                                                           use_case_name=USE_CASE_NAME,
                                                                           features_df=FEATURES_DF,
                                                                           selected_features=SELECTED_FEATURES,
                                                                           selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                                           selected_k_means_count=SELECTED_KMEANS_CLUSTERS_COUNT,
                                                                           save_to_file=SELECT_SAVE_FILES)

    # DBSCAN: sh_score
    clustering.plot_sh_score_DBSCAN(use_this_function=False,  # True | False
                                    selected_cohort=SELECTED_COHORT_preprocessed,
                                    cohort_title=SELECTED_COHORT_TITLE,
                                    use_case_name=USE_CASE_NAME,
                                    features_df=FEATURES_DF,
                                    selected_features=SELECTED_FEATURES,
                                    selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                    save_to_file=SELECT_SAVE_FILES)
    # DBSCAN: plotting
    SELECTED_EPS = 0.5  # manually checking silhouette score shows optimal epsilon-min_sample-combination
    SELECTED_MIN_SAMPLE = 5
    clustering.plot_DBSCAN_on_pacmap(use_this_function=False,  # True | False
                                     selected_cohort=SELECTED_COHORT_preprocessed,
                                     cohort_title=SELECTED_COHORT_TITLE,
                                     use_case_name=USE_CASE_NAME,
                                     features_df=FEATURES_DF,
                                     selected_features=SELECTED_FEATURES,
                                     selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                     selected_eps=SELECTED_EPS,
                                     selected_min_sample=SELECTED_MIN_SAMPLE,
                                     save_to_file=SELECT_SAVE_FILES)


    ### Machine Learning Predictions
    # Step 4.1) Classification: (RandomForest, XGBoost, ...)
    SELECTED_CLASSIFICATION_METHOD = 'XGBoost'  # options: RandomForest | XGBoost
    SELECTED_SAMPLING_METHOD = 'oversampling'  # options: no_sampling | oversampling | undersampling   -> estimation: oversampling > no_sampling > undersampling (very bad results)
    ALL_CLASSIFICATION_METHODS: list = ['RandomForest', 'XGBoost']
    # Confusion Matrix (CM)
    cm = classification.get_confusion_matrix(use_this_function=False,  # True | False
                                             classification_method=SELECTED_CLASSIFICATION_METHOD,
                                             sampling_method=SELECTED_SAMPLING_METHOD,
                                             selected_cohort=SELECTED_COHORT_preprocessed,
                                             cohort_title=SELECTED_COHORT_TITLE,
                                             use_case_name=USE_CASE_NAME,
                                             features_df=FEATURES_DF,
                                             selected_features=SELECTED_FEATURES,
                                             selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                             verbose=True,
                                             save_to_file=SELECT_SAVE_FILES
                                             )
    # Classification Report
    report = classification.get_classification_report(use_this_function=False,  # True | False
                                                      classification_method=SELECTED_CLASSIFICATION_METHOD,
                                                      sampling_method=SELECTED_SAMPLING_METHOD,
                                                      selected_cohort=SELECTED_COHORT_preprocessed,
                                                      cohort_title=SELECTED_COHORT_TITLE,
                                                      use_case_name=USE_CASE_NAME,
                                                      features_df=FEATURES_DF,
                                                      selected_features=SELECTED_FEATURES,
                                                      selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                      verbose=True,
                                                      save_to_file=SELECT_SAVE_FILES
                                                      )
    # AUROC (plot & score)
    auc_score = classification.get_auc_score(use_this_function=False,  # True | False
                                             classification_method=SELECTED_CLASSIFICATION_METHOD,
                                             sampling_method=SELECTED_SAMPLING_METHOD,
                                             selected_cohort=SELECTED_COHORT_preprocessed,
                                             cohort_title=SELECTED_COHORT_TITLE,
                                             use_case_name=USE_CASE_NAME,
                                             features_df=FEATURES_DF,
                                             selected_features=SELECTED_FEATURES,
                                             selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                             show_plot=True,
                                             verbose=True,
                                             save_to_file=SELECT_SAVE_FILES
                                             )

    # Step 4.2) Deep Learning/Neural Network (might also be inside 4.1? Or only useful if based on timeseries?)
    # ...

    # Step 4.3) Models Overview -> table to compare all accuracy & recall results
    classification.compare_classification_models_on_cohort(use_this_function=False,  # True | False
                                                           use_case_name=USE_CASE_NAME,
                                                           features_df=FEATURES_DF,
                                                           selected_features=SELECTED_FEATURES,
                                                           all_cohorts_with_titles=ALL_COHORTS_WITH_TITLES,
                                                           all_classification_methods=ALL_CLASSIFICATION_METHODS,
                                                           all_dependent_variables=ALL_DEPENDENT_VARIABLES,
                                                           save_to_file=SELECT_SAVE_FILES)

    # Input: selected_cohort and its ideal kmeans cluster-count
    # Output: table of prediction quality per cluster, rows = different model configs (per classification_method and dependent_variable)
    classification.compare_classification_models_on_clusters(use_this_function=True,  # True | False
                                                             use_case_name=USE_CASE_NAME,
                                                             features_df=FEATURES_DF,
                                                             selected_features=SELECTED_FEATURES,
                                                             selected_cohort=SELECTED_COHORT_preprocessed,
                                                             all_classification_methods=ALL_CLASSIFICATION_METHODS,
                                                             all_dependent_variables=ALL_DEPENDENT_VARIABLES,
                                                             selected_k_means_count=SELECTED_KMEANS_CLUSTERS_COUNT,
                                                             check_sh_score=True,
                                                             save_to_file=SELECT_SAVE_FILES)

    ### Fairness Metrics
    # Step 5.1) Calculate Fairness for manual Subgroups

    ### Automated Subgroup detection
    # Step 6.1) Calculate automated Subgroups and related fairness metrics

    # Step 6.2) Include ASDF-Dashboard as frontend
