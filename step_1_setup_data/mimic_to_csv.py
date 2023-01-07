import csv
import os
import time
from datetime import datetime
from os.path import isfile, join

import psycopg2

from step_1_setup_data import SQL_queries
from step_1_setup_data.db_setup import config
from supplements import selection_comorbidities_icd9


def export_patients_to_csv(project_path: str, use_case_icd_list=None, use_case_itemids=None, use_case_name=None) -> None:
    """
    This function exports a .csv file per unique-admission (each patient once).
    The files are saved inside /export/use_case_name/
    The function does not return anything.
    Analysis of the selection can be conducted on the .csv file created by export_basic_statistics
    :param project_path: local directory where files are stored, must be defined by the user
    :param use_case_icd_list: recommended to use this list of all icd_9 codes that should be selected for this use-case
    :param use_case_itemids:
    :param use_case_name: the title for this use-case e.g. stroke, heart_failure, sepsis
    :return: None
    """
    if project_path is None:
        print('ERROR: project_path must be defined by the user.')
        return None
    if use_case_icd_list is None:
        use_case_icd_list = []
        use_case_name = 'all_cases'
        print('NOTICE: No icd9 codes were selected. Not recommended, but query will be run for all available use-cases.')
    if use_case_itemids is None:
        use_case_itemids = []
        print('NOTICE: No itemids were selected. Query will be run for all available itemids.')
    if use_case_name is None:
        use_case_name = 'no_use_case_name'
        print('NOTICE: No use-case name was chosen. Export files will be saved in folder "no_use_case_name".')

    # Setup itemids Filter
    if len(use_case_itemids) == 0:
        selected_itemids_string: str = '\'{}\''
    else:
        selected_itemids_string: str = '\'{'
        for itemid in use_case_itemids:
            selected_itemids_string = selected_itemids_string + str(itemid) + ', '
        selected_itemids_string = selected_itemids_string[:-2] + '}\''
    # print('CHECK: itemid Filter:', len(use_case_itemids))

    # Setup connection to PostGre MIMIC-III Database
    db_params: dict = config()
    conn = None
    try:
        ##### 0) Connect to the database
        conn = psycopg2.connect(**db_params)
        cursor_1 = conn.cursor()
        print('STATUS: Connection to the PostgreSQL database successful.')

        ##### 1) Execute the query_patient_cohort
        patient_cohort: list = SQL_queries.query_patient_cohort(cursor_1, use_case_icd_list)
        cohort_header: list = SQL_queries.query_header_patient_cohort(cursor_1)

        # create new directory for this use-case if it does not exist yet
        directory: str = f'{project_path}exports/{use_case_name}'
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        # export to csv
        filename_string: str = f'{directory}/0_patient_cohort.csv'
        filename = filename_string.encode()
        with open(filename, 'w', newline='') as output_file:                        # use newline with windows
            csv_out = csv.writer(output_file)
            csv_out.writerow(cohort_header)
            csv_out.writerows(patient_cohort)
        print('STATUS: 0_patient_cohort.csv created.')


        ##### 2) export chart_events data for each icu_stay #####
        # Get icu_stay_ids from patient_cohort
        icu_stay_ids: list = []
        icu_stay_ids_set: set = set()
        for entry in patient_cohort:
            icu_stay_ids.append(entry[0])
            icu_stay_ids_set.add(entry[0])
        # icu_stay_ids.sort()
        if len(icu_stay_ids) != len(icu_stay_ids_set):
            print('Warning: Duplicate icustay_ids exist. Recommended to check patient_cohort.')

        # Get chart_events for each icustay and export to .csv
        query_counter = 0
        seconds_cumulated = 0
        for icustay_id in icu_stay_ids[:150]:             # todo reminder: loop through for all ids, also turn on sorting again
            print('STATUS: Executing query_single_icustay for icustay_id', str(icustay_id))
            query_counter += 1
            starting_time = datetime.now()
            single_icustay: list = SQL_queries.query_single_icustay(cursor_1, icustay_id, selected_itemids_string)
            # print('CHECK: Query result:', single_icustay)
            remaining_queries: int = len(icu_stay_ids) - query_counter
            single_header: list = SQL_queries.query_header_single(cursor_1)
            time_needed = datetime.now() - starting_time
            seconds_cumulated += time_needed.total_seconds()
            avg_seconds = seconds_cumulated / query_counter
            print('CHECK: Seconds needed for last Query:', round(time_needed.total_seconds()))
            print(f'CHECK: Estimated minutes for remaining {remaining_queries} Queries: {round(avg_seconds * remaining_queries / 60)}')

            filename_string: str = f'{directory}/icustay_id_{icustay_id}.csv'
            filename = filename_string.encode()
            with open(filename, 'w', newline='') as output_file:                    # use newline with windows
                csv_out = csv.writer(output_file)
                csv_out.writerow(single_header)
                csv_out.writerows(single_icustay)

        # cursor_1.execute('DROP TABLE public.temp_filtered_patient_cohort;')       # deleting table temp_filtered_patient_cohort, it should always only exist in DB for the current use-case
        conn.commit()
        cursor_1.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)

    finally:
        if conn is not None:
            conn.close()
        print('STATUS: Finished mimic_to_csv.')


### SETUP FUNCTIONS ###
def create_table_all_diagnoses() -> None:
    """
    This function is only needed when using the database for the first time.
    It creates the table 'all_diagnoses_icd' where for each admission all available diagnoses are saved in the new
    field 'all_icd_codes' as an array.
    Thus, in later filtering for only one admission, all other diagnoses for this admission are not lost.
    """
    # Setup connection to PostGre MIMIC-III Database
    db_params: dict = config()
    conn = None
    try:
        # connect to the database
        print('Connecting to the PostgreSQL database for create_table_all_diagnoses:')
        conn = psycopg2.connect(**db_params)
        cur_1 = conn.cursor()
        print('Connection successful.')
        ##### create the necessary table 'all_diagnoses_icd' #####
        SQL_queries.query_create_table_all_diagnoses_icd(cur_1)
        # close the communication with the database
        conn.commit()
        cur_1.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)
    finally:
        if conn is not None:
            conn.close()


def setup_postgre_files():
    """
    This function is only needed when using the database for the first time.
    It creates all necessary backend files (functions and views) for the postgre database.
    Sometimes does not create the functions. Then it is easiest to simply copy&paste the SQL Scripts into
    Postgre QueryTool and execute each there.

    IMPORTANT: The OASIS-Score scripts in /supplements/SQL/Dayly-SAPS-III-and-OASIS-scores-for-MIMIC-III-master
    must also be loaded into the postgres DB manually.
    """
    # Setup connection to PostGre MIMIC-III Database
    db_params: dict = config()
    conn = None
    try:
        # connect to the database
        print('Connecting to the PostgreSQL database for setup_postgre_files:')
        conn = psycopg2.connect(**db_params)
        cur_1 = conn.cursor()
        print('Connection successful.')
        ##### run query #####
        for file in os.listdir('supplements/SQL/'):
            if isfile(join('supplements/SQL/', file)):  # making sure only files, no other dictionaries inside the SQL folder
                if not file == 'create_events_dictionary.sql' and not file == 'create_icd9_codes_dictionary.sql':  # dictionary creation in other function (is optional, takes 30 seconds)
                    query_setup_postgre_files_string: str = open(f'supplements/SQL/{file}', 'r').read()
                    print('STATUS: Loading file into postgre database:', file)
                    cur_1.execute(query_setup_postgre_files_string)
                    time.sleep(2)  # safety to give db time to create files
        print('STATUS: query_setup_postgre_files_string executed.')
        # close the communication with the database
        conn.commit()
        cur_1.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)
    finally:
        if conn is not None:
            conn.close()


def create_supplement_dictionaries():
    """
    This function is optional to create the supplement dictionary files.
    They can also be created by manually copying & pasting the 'create_events_dictionary.sql' script into postgres.
    """
    # Setup connection to PostGre MIMIC-III Database
    db_params: dict = config()
    conn = None
    try:
        # connect to the database
        print('Connecting to the PostgreSQL database for setup_postgre_files:')
        conn = psycopg2.connect(**db_params)
        cur_1 = conn.cursor()
        print('Connection successful.')

        # run create_dictionaries query
        for file in os.listdir('supplements/SQL/'):
            if isfile(join('supplements/SQL/',
                           file)):  # making sure only files, no other dictionaries inside the SQL folder
                if file == 'create_events_dictionary.sql':  # dictionary creation in other function (is optional, takes 30 seconds)
                    query_create_label_dictionaries_string: str = open(f'supplements/SQL/{file}', 'r').read()
                    print('STATUS: Loading file into postgre database:', file)
                    cur_1.execute(query_create_label_dictionaries_string)
                    events_dictionary = cur_1.fetchall()

                    filename_string: str = './supplements/events_dictionary.csv'
                    filename = filename_string.encode()
                    with open(filename, 'w', newline='') as output_file:
                        csv_out = csv.writer(output_file)
                        csv_out.writerow(['itemid', 'label', 'category', 'dbsource', 'valueuom'])
                        csv_out.writerows(events_dictionary)
                    time.sleep(2)

                if file == 'create_icd9_codes_dictionary.sql':
                    query_create_label_dictionaries_string: str = open(f'supplements/SQL/{file}', 'r').read()
                    print('STATUS: Loading file into postgre database:', file)
                    cur_1.execute(query_create_label_dictionaries_string)
                    events_dictionary = cur_1.fetchall()

                    filename_string: str = './supplements/icd9_codes_dictionary.csv'
                    filename = filename_string.encode()
                    with open(filename, 'w', newline='') as output_file:
                        csv_out = csv.writer(output_file)
                        csv_out.writerow(['icd9_code', 'short_title', 'long_title', 'source'])
                        csv_out.writerows(events_dictionary)
                    time.sleep(2)  # safety to give db time to create files
        print('STATUS: query_create_label_dictionaries executed.')

        # close the communication with the database
        conn.commit()
        cur_1.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)
    finally:
        if conn is not None:
            conn.close()


def load_comorbidities_into_db():
    """
        This function loads the lists selected in supplements/selection_comorbidities_icd9.py into a table in the
        postgres DB. This table will be used within the creation of the patient.csv files to derive
        possible comorbidities.
        """
    # Setup connection to PostGre MIMIC-III Database
    db_params: dict = config()
    conn = None
    try:
        # connect to the database
        print('Connecting to the PostgreSQL database for load_comorbidities_into_db:')
        conn = psycopg2.connect(**db_params)
        cur_1 = conn.cursor()
        print('Connection successful.')
        ##### run query #####
        cur_1.execute('SET search_path TO public;')
        cur_1.execute('DROP TABLE IF EXISTS comorbidity_codes;')
        cur_1.execute('CREATE TABLE comorbidity_codes (codes varchar(10), category varchar(20));')

        # Loading sepsis_list
        for icd9_code in selection_comorbidities_icd9.sepsis_list:
            cur_1.execute(f'INSERT INTO public.comorbidity_codes (codes, category) VALUES (\'{icd9_code}\', \'sepsis\');')
        print('STATUS: Finished loading sepsis_list into comorbidity_codes table.')
        # Loading hypertension_list
        for icd9_code in selection_comorbidities_icd9.hypertension_list:
            cur_1.execute(f'INSERT INTO public.comorbidity_codes (codes, category) VALUES (\'{icd9_code}\', \'hypertension\');')
        print('STATUS: Finished loading hypertension_list into comorbidity_codes table.')
        # Loading obesity_list
        for icd9_code in selection_comorbidities_icd9.obesity_list:
            cur_1.execute(
                f'INSERT INTO public.comorbidity_codes (codes, category) VALUES (\'{icd9_code}\', \'obesity\');')
        print('STATUS: Finished loading obesity_list into comorbidity_codes table.')
        # Loading diabetes_list
        for icd9_code in selection_comorbidities_icd9.diabetes_list:
            cur_1.execute(
                f'INSERT INTO public.comorbidity_codes (codes, category) VALUES (\'{icd9_code}\', \'diabetes\');')
        print('STATUS: Finished loading diabetes_list into comorbidity_codes table.')
        # Loading drug_abuse_list
        for icd9_code in selection_comorbidities_icd9.drug_abuse_list:
            cur_1.execute(
                f'INSERT INTO public.comorbidity_codes (codes, category) VALUES (\'{icd9_code}\', \'drug_abuse\');')
        print('STATUS: Finished loading drug_abuse_list into comorbidity_codes table.')
        # Loading cancer_list
        for icd9_code in selection_comorbidities_icd9.cancer_list:
            cur_1.execute(
                f'INSERT INTO public.comorbidity_codes (codes, category) VALUES (\'{icd9_code}\', \'cancer\');')
        print('STATUS: Finished loading cancer_list into comorbidity_codes table.')

        # close the communication with the database
        time.sleep(0.5)
        conn.commit()
        cur_1.close()
        print('STATUS: Finished creating comorbidity_codes table.')
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)
    finally:
        if conn is not None:
            conn.close()
