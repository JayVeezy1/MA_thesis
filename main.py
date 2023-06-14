from datetime import datetime
import pandas as pd

from objects.patients import Patient
from step_1_setup_data import cache_IO
from step_2_preprocessing import preprocessing_functions
from step_2_preprocessing.preprocessing_functions import get_preprocessed_avg_cohort
from step_3_data_analysis import correlations, clustering, general_statistics, data_visualization
from step_4_classification import classification_deeplearning, classification
from step_5_fairness import fairness_analysis
from step_6_subgroup_analysis import subgroup_analysis
from web_app.util import start_streamlit_frontend

####### MAIN #######
# By: Jakob Vanek, 2023, Master Thesis at Goethe University
if __name__ == '__main__':
    starting_time = datetime.now()
    PROJECT_PATH: str = 'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'  # this variable must be fitted to the users local project folder
    PROJECT_PATH_LAPTOP = 'C:/Users/vanek/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/'
    # PROJECT_PATH = PROJECT_PATH_LAPTOP
    USE_CASE_NAME: str = 'stroke_all_systems'
    FEATURES_DF = pd.read_excel('./supplements/FEATURE_PREPROCESSING_TABLE.xlsx')
    SELECTED_DEPENDENT_VARIABLE = 'death_in_hosp'
    ALL_DEPENDENT_VARIABLES: list = ['death_in_hosp', 'death_3_days']   # , 'death_30_days', 'death_180_days', 'death_365_days']
    ### Setup, MIMIC-III Export from DB, Load from Cache
    # Step 0) Setup when first time using db:
    # mimic_to_csv.setup_postgre_files()                 # setup all needed background functions and views for postgre. Warning: Sometimes this setup from Python does not work. Then you simply copy&paste each SQL Script into PostGre QueryTool and execute it.
    # mimic_to_csv.create_table_all_diagnoses()          # create a necessary table 'all_diagnoses' where for each admission all available diagnoses are saved in the new field 'all_icd_codes' (takes approx. 2 hours)
    # mimic_to_csv.create_supplement_dictionaries()      # create supplementary dictionary files
    # mimic_to_csv.load_comorbidities_into_db()          # create the necessary table 'comorbidity_codes' where the icd9_codes that are used to find important comorbidities are loaded into the DB
    # Step 1.1) Export the raw patient data for the specified use_case (icd_list) into .csv files, all available features will be exported
    # metavision stroke use-case has 1232 patients, each takes approx. 30 seconds -> 500 Minutes, 8,5 hours
    # complete stroke cases has 2655 -> 1300 minutes, 20 hours
    # Run this function only once for the patient-export. Afterward use .csvs
    # mimic_to_csv.export_patients_to_csv(project_path=PROJECT_PATH,
    #                                   use_case_icd_list=selection_icd9_codes.selected_stroke_codes,            # stroke case = icd9_00_stroke_selected
    #                                  use_case_itemids=[],
    #                                 use_case_name=USE_CASE_NAME)
    # Step 1.2) Filter final patient.csvs for relevant features and export as 'final_dataset'
    # transform raw.csvs into filtered, final .csvs, also transform carevue feature-names into metavision names
    # select_relevant_features.export_final_dataset(project_path=PROJECT_PATH, use_case_name=USE_CASE_NAME)

    # Step 1.3) Load all .csv files as a 'Patient' Object, use Pickle for Cache
    DELETE_CACHE = False
    SELECT_SAVE_FILES = False
    cache_IO.load_data_from_cache(project_path=PROJECT_PATH, features_df=FEATURES_DF, use_case_name=USE_CASE_NAME,
                                  delete_existing_cache=DELETE_CACHE)
    ### Preprocessing
    # Step 2) Calculate Avg, Filter, Scale, Impute & Interpolate for each patient
    raw_avg_cohort = Patient.get_avg_patient_cohort(project_path=PROJECT_PATH, use_case_name=USE_CASE_NAME,
                                                    features_df=FEATURES_DF, delete_existing_cache=DELETE_CACHE,
                                                    selected_patients=[])   # empty=all
    # Scaling
    SELECTED_COHORT = Patient.get_avg_scaled_data(raw_avg_cohort, FEATURES_DF)
    # Choose Cohort Parameters
    ALL_DATABASES: list = ['complete']      # , 'metavision', 'carevue']
    ALL_STROKE_TYPES: list = ['all_stroke']     # , 'hemorrhagic', 'ischemic', 'other_stroke']
    SELECTED_DATABASE = 'complete'
    SELECTED_STROKE_TYPE = 'all_stroke'
    SELECTED_COHORT_TITLE = 'scaled_' + SELECTED_DATABASE + '_avg_cohort_' + SELECTED_STROKE_TYPE
    SELECTED_COHORT_preprocessed = get_preprocessed_avg_cohort(avg_cohort=SELECTED_COHORT,
                                                               features_df=FEATURES_DF,
                                                               selected_database=SELECTED_DATABASE,
                                                               selected_stroke_type=SELECTED_STROKE_TYPE)
    SELECTED_FEATURES = list(SELECTED_COHORT_preprocessed.columns)

    # Automated: List of all cohorts_preprocessed for model comparison
    ALL_COHORTS_WITH_TITLES: dict = preprocessing_functions.get_all_cohorts(SELECTED_COHORT, FEATURES_DF, SELECTED_DATABASE)
    print('STATUS: Preprocessing finished.\n')


    # todo text: update subgroup chapter
    ## Check error with fairness measure calculation for cluster 8

    # todo code: add SHAPley values to classification page
    ## probably choose different explainer objects depending on selected prediction model
    ## add shapley to classification chapter and compare with correlation (mention future work: different importance per subgroups)


    ### Data Analysis
    # Step 3.1) General Statistics
    general_statistics.calculate_feature_overview_table(use_this_function=False,  # True | False
                                                        selected_cohort=SELECTED_COHORT_preprocessed,
                                                        features_df=FEATURES_DF,
                                                        selected_features=SELECTED_FEATURES,
                                                        cohort_title=SELECTED_COHORT_TITLE,
                                                        use_case_name=USE_CASE_NAME,
                                                        selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                        save_to_file=SELECT_SAVE_FILES)

    general_statistics.calculate_deaths_table(use_this_function=False,  # True | False
                                              selected_cohort=SELECTED_COHORT_preprocessed,
                                              cohort_title=SELECTED_COHORT_TITLE,
                                              use_case_name=USE_CASE_NAME,
                                              save_to_file=SELECT_SAVE_FILES)

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
                                      use_encoding=True,
                                      save_to_file=SELECT_SAVE_FILES)

    # Step 3.4) Clustering (kmeans, kprototype, DBSCAN, ...)
    # KMEANS
    SELECTED_KMEANS_CLUSTERS_COUNT = 3  # manually checking silhouette score shows optimal cluster count (higher is better)
    clustering.plot_k_means_on_pacmap(use_this_function=False,  # True | False
                                      display_sh_score=False,  # option for sh_score
                                      selected_cohort=SELECTED_COHORT_preprocessed,
                                      cohort_title=SELECTED_COHORT_TITLE,
                                      use_case_name=USE_CASE_NAME,
                                      features_df=FEATURES_DF,
                                      selected_features=SELECTED_FEATURES,
                                      selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                      selected_cluster_count=SELECTED_KMEANS_CLUSTERS_COUNT,
                                      use_encoding=True,
                                      save_to_file=SELECT_SAVE_FILES)

    # KPrototypes
    SELECTED_KPROTO_CLUSTERS_COUNT = 5
    clustering.plot_k_prot_on_pacmap(use_this_function=False,  # True | False
                                     display_sh_score=False,  # option for sh_score
                                     selected_cohort=SELECTED_COHORT_preprocessed,
                                     cohort_title=SELECTED_COHORT_TITLE,
                                     use_case_name=USE_CASE_NAME,
                                     features_df=FEATURES_DF,
                                     selected_features=SELECTED_FEATURES,
                                     selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                     selected_cluster_count=SELECTED_KPROTO_CLUSTERS_COUNT,
                                     use_encoding=False,  # not needed for kPrototypes
                                     save_to_file=SELECT_SAVE_FILES)

    # DBSCAN
    SELECTED_EPS = 0.7  # manually checking silhouette score shows optimal epsilon-min_sample-combination
    SELECTED_MIN_SAMPLE = 5
    clustering.plot_DBSCAN_on_pacmap(use_this_function=False,  # True | False
                                     display_sh_score=False,  # option for sh_score
                                     selected_cohort=SELECTED_COHORT_preprocessed,
                                     cohort_title=SELECTED_COHORT_TITLE,
                                     use_case_name=USE_CASE_NAME,
                                     features_df=FEATURES_DF,
                                     selected_features=SELECTED_FEATURES,
                                     selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                     selected_eps=SELECTED_EPS,
                                     selected_min_sample=SELECTED_MIN_SAMPLE,
                                     use_encoding=True,
                                     save_to_file=SELECT_SAVE_FILES)

    # SLINK
    SELECTED_CRITERION = 'distance'     # options: inconsistent | maxclust | monocrit
    SELECTED_THRESHOLD = 1.41
    clustering.plot_SLINK_on_pacmap(use_this_function=False,
                                    display_sh_score=False,
                                    selected_cohort=SELECTED_COHORT_preprocessed,
                                    cohort_title=SELECTED_COHORT_TITLE,
                                    use_case_name=USE_CASE_NAME,
                                    features_df=FEATURES_DF,
                                    selected_features=SELECTED_FEATURES,
                                    selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                    use_encoding=False,
                                    show_dendrogram=False,
                                    separation_criterion=SELECTED_CRITERION,
                                    threshold=SELECTED_THRESHOLD,
                                    save_to_file=SELECT_SAVE_FILES)

    ### Machine Learning Predictions
    # Step 4.1) Classification: (RandomForest, XGBoost, ...)
    SELECTED_CLASSIFICATION_METHOD = 'XGBoost'  # options: RandomForest | XGBoost || NOT deeplearning_sequential -> use function get_classification_report_deeplearning()
    USE_GRIDSEARCH = True
    SELECTED_SAMPLING_METHOD = 'oversampling'  # options: no_sampling | oversampling | undersampling   -> estimation: oversampling > no_sampling > undersampling (very bad results)
    ALL_CLASSIFICATION_METHODS: list = ['RandomForest', 'RandomForest_with_gridsearch', 'XGBoost', 'deeplearning_sequential']
    # Classification Report
    report = classification.get_classification_report(use_this_function=False,  # True | False
                                                      display_confusion_matrix=True,  # option for CM
                                                      classification_method=SELECTED_CLASSIFICATION_METHOD,
                                                      sampling_method=SELECTED_SAMPLING_METHOD,
                                                      selected_cohort=SELECTED_COHORT_preprocessed,
                                                      cohort_title=SELECTED_COHORT_TITLE,
                                                      use_case_name=USE_CASE_NAME,
                                                      features_df=FEATURES_DF,
                                                      selected_features=SELECTED_FEATURES,
                                                      selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                      use_grid_search=USE_GRIDSEARCH,
                                                      verbose=True,
                                                      save_to_file=SELECT_SAVE_FILES)
    # AUROC + AUPRC (plot & score)
    auc_score, auroc_plot, auc_prc_score, auc_prc_plot = classification.get_auc_score(use_this_function=False,  # True | False
                                                            classification_method=SELECTED_CLASSIFICATION_METHOD,
                                                            sampling_method=SELECTED_SAMPLING_METHOD,
                                                            selected_cohort=SELECTED_COHORT_preprocessed,
                                                            cohort_title=SELECTED_COHORT_TITLE,
                                                            use_case_name=USE_CASE_NAME,
                                                            features_df=FEATURES_DF,
                                                            selected_features=SELECTED_FEATURES,
                                                            selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                            show_plot=True,
                                                            use_grid_search=USE_GRIDSEARCH,
                                                            verbose=True,
                                                            save_to_file=SELECT_SAVE_FILES)

    # Plot optimal RandomForest (based on GridSearchCV)
    forest_plot = classification.plot_random_forest(use_this_function=False,  # True | False
                                                    classification_method='RandomForest',
                                                    sampling_method=SELECTED_SAMPLING_METHOD,
                                                    selected_cohort=SELECTED_COHORT_preprocessed,
                                                    cohort_title=SELECTED_COHORT_TITLE,
                                                    use_case_name=USE_CASE_NAME,
                                                    features_df=FEATURES_DF,
                                                    selected_features=SELECTED_FEATURES,
                                                    selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                    show_plot=True,
                                                    use_grid_search=USE_GRIDSEARCH,
                                                    verbose=True,
                                                    save_to_file=SELECT_SAVE_FILES)

    # Step 4.2) Classification Report Deep Learning Neural Network
    report_DL = classification_deeplearning.get_classification_report_deeplearning(use_this_function=False,
                                                                                   # True | False
                                                                                   sampling_method=SELECTED_SAMPLING_METHOD,
                                                                                   selected_cohort=SELECTED_COHORT_preprocessed,
                                                                                   cohort_title=SELECTED_COHORT_TITLE,
                                                                                   use_case_name=USE_CASE_NAME,
                                                                                   features_df=FEATURES_DF,
                                                                                   selected_features=SELECTED_FEATURES,
                                                                                   selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                                                   verbose=True,
                                                                                   save_to_file=SELECT_SAVE_FILES)

    # Step 4.3) Models Overview -> table to compare all accuracy & recall results
    classification.compare_classification_models_on_cohort(use_this_function=False,  # True | False
                                                           use_case_name=USE_CASE_NAME,
                                                           features_df=FEATURES_DF,
                                                           selected_features=SELECTED_FEATURES,
                                                           sampling_method=SELECTED_SAMPLING_METHOD,
                                                           all_cohorts_with_titles=ALL_COHORTS_WITH_TITLES,
                                                           all_classification_methods=ALL_CLASSIFICATION_METHODS,
                                                           all_dependent_variables=ALL_DEPENDENT_VARIABLES,
                                                           save_to_file=SELECT_SAVE_FILES)

    # Step 4.4) Shapley Feature Relevance
    classification.get_shapely_relevance(use_this_function=False,
                                         selected_feature='oasis',      # ethnicity_1
                                         classification_method=SELECTED_CLASSIFICATION_METHOD,
                                         sampling_method=SELECTED_SAMPLING_METHOD,
                                         selected_cohort=SELECTED_COHORT_preprocessed,
                                         cohort_title=SELECTED_COHORT_TITLE,
                                         use_case_name=USE_CASE_NAME,
                                         features_df=FEATURES_DF,
                                         selected_features=SELECTED_FEATURES,
                                         selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                         show_plot=True,
                                         use_grid_search=USE_GRIDSEARCH,
                                         verbose=True,
                                         save_to_cache=False,
                                         save_to_file=SELECT_SAVE_FILES)

    ### Fairness Metrics
    # Step 5.1) Calculate Fairness for manual Subgroups
    PROTECTED_FEATURES = ['gender']         # 'ethnicity',
    PRIVILEGED_VALUES = [['F']]      # , ['ASIAN'],  'WHITE'
    fairness_analysis.get_fairness_report(use_this_function=False,  # True | False
                                          plot_performance_metrics=False,
                                          classification_method=SELECTED_CLASSIFICATION_METHOD,
                                          sampling_method=SELECTED_SAMPLING_METHOD,
                                          selected_cohort=SELECTED_COHORT_preprocessed,
                                          cohort_title=SELECTED_COHORT_TITLE,
                                          use_case_name=USE_CASE_NAME,
                                          features_df=FEATURES_DF,
                                          selected_features=SELECTED_FEATURES,
                                          selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                          verbose=True,
                                          protected_features=PROTECTED_FEATURES,
                                          privileged_values=PRIVILEGED_VALUES,
                                          use_grid_search=USE_GRIDSEARCH,
                                          save_to_file=SELECT_SAVE_FILES)

    ### Subgroup Analysis
    # Step 6.1) Calculate automated Subgroups and related fairness metrics -> Inside ASDF-Dashboard
    # Cluster Comparison/Subgroup Detection
    subgroups_overview = subgroup_analysis.derive_subgroups(use_this_function=False,     # True | False
                                                            selected_cohort=SELECTED_COHORT_preprocessed,
                                                            cohort_title=SELECTED_COHORT_TITLE,
                                                            use_case_name=USE_CASE_NAME,
                                                            features_df=FEATURES_DF,
                                                            selected_features=SELECTED_FEATURES,
                                                            selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                            selected_cluster_count=SELECTED_KMEANS_CLUSTERS_COUNT,
                                                            clustering_method='kmeans',
                                                            use_encoding=True,
                                                            save_to_file=SELECT_SAVE_FILES)

    # Feature Influence per Cluster
    feature_influence = subgroup_analysis.calculate_feature_influence_table(use_this_function=False,  # True | False
                                                                            selected_cohort=SELECTED_COHORT_preprocessed,
                                                                            cohort_title=SELECTED_COHORT_TITLE,
                                                                            use_case_name=USE_CASE_NAME,
                                                                            features_df=FEATURES_DF,
                                                                            selected_features=SELECTED_FEATURES,
                                                                            selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                                            selected_cluster_count=SELECTED_KMEANS_CLUSTERS_COUNT,
                                                                            selected_cluster='all',    # can be selected in frontend
                                                                            show_value_influences=True,
                                                                            clustering_method='kmeans',
                                                                            use_encoding=True,
                                                                            save_to_file=SELECT_SAVE_FILES)

    # Classification Comparison over Clusters (Fairness Pre-Check)
    subgroup_analysis.compare_classification_models_on_clusters(use_this_function=False,  # True | False
                                                                use_case_name=USE_CASE_NAME,
                                                                features_df=FEATURES_DF,
                                                                selected_features=SELECTED_FEATURES,
                                                                selected_cohort=SELECTED_COHORT_preprocessed,
                                                                cohort_title=SELECTED_COHORT_TITLE,
                                                                classification_method=SELECTED_CLASSIFICATION_METHOD,
                                                                sampling_method=SELECTED_SAMPLING_METHOD,
                                                                dependent_variable=SELECTED_DEPENDENT_VARIABLE,
                                                                clustering_method='kmeans',
                                                                selected_cluster_count=SELECTED_KMEANS_CLUSTERS_COUNT,
                                                                use_grid_search=True,
                                                                check_sh_score=True,
                                                                use_encoding=True,
                                                                save_to_file=SELECT_SAVE_FILES)

    ### Step 7.1) Streamlit App for Visualization
    start_streamlit_frontend(use_this_function=True)

    ### Deprecated: ASDF-Dashboard for visualization  https://github.com/jeschaef/ASDF-Dashboard
    # Important: Start Background Services First
    # Redis: docker run --name redis -p 6379:6379 -d redis (once created 'start' in Docker Desktop)
    # Celery (in second cmd terminal): celery -A frontend.app.celery_app worker -P solo -l info
    # app = start_dashboard_from_main(use_this_function=True)
    # Needed for upload: use following function to create dataset for original fairness visualizations of the ASDF-Dashboard
    # cohort_classified = classification.get_cohort_classified(use_this_function=False,  # True | False
    #                                                          project_path=PROJECT_PATH,  # to save where avg_cohort is
    #                                                          classification_method=SELECTED_CLASSIFICATION_METHOD,
    #                                                          sampling_method=SELECTED_SAMPLING_METHOD,
    #                                                          selected_cohort=SELECTED_COHORT_preprocessed,
    #                                                          cohort_title=SELECTED_COHORT_TITLE,
    #                                                          use_case_name=USE_CASE_NAME,
    #                                                          features_df=FEATURES_DF,
    #                                                          selected_features=SELECTED_FEATURES,
    #                                                          selected_dependent_variable=SELECTED_DEPENDENT_VARIABLE,
    #                                                          use_grid_search=USE_GRIDSEARCH,
    #                                                          verbose=True,
    #                                                          save_to_file=True)

    print(f'\nSTATUS: Analysis finished for {len(SELECTED_FEATURES)} selected_features.')
    time_diff = datetime.now() - starting_time
    print(f'STATUS: Seconds needed: ', time_diff.total_seconds())
