import csv
import os
from datetime import time, datetime

import psycopg2

import SQL_queries
from db_setup import config


def export_patients_to_csv(use_case_icd_list=None, use_case_name=None) -> None:
    """
    This function exports a .csv file per unique-admission (each patient once).
    The files are saved inside /export/use_case_name/
    The function does not return anything.
    Analysis of the selection can be conducted on the .csv file created by export_basic_statistics
    :param use_case_icd_list: recommended to use this list of all icd_9 codes that should be selected for this use-case
    :param use_case_name: the title for this use-case e.g. stroke, heart_failure, sepsis
    :return: None
    """
    if use_case_icd_list is None:
        use_case_icd_list = []
        use_case_name = 'all_cases'
    if use_case_name is None:
        use_case_name = 'no_use_case_name'

    # Setup connection to PostGre MIMIC-III Database
    db_params: dict = config()

    conn = None
    try:
        # connect to the database
        conn = psycopg2.connect(**db_params)
        cursor_1 = conn.cursor()
        print('STATUS: Connection to the PostgreSQL database successful.')

        ##### 1) execute the query_patient_cohort
        patient_cohort: list = SQL_queries.query_patient_cohort(cursor_1, use_case_icd_list)
        cohort_header: list = SQL_queries.query_header_patient_cohort(cursor_1)

        # create new directory for this use-case if it does not exist yet
        directory: str = f'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/exports/{use_case_name}'
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        # export to csv
        filename_string: str = f'{directory}/0_patient_cohort.csv'
        filename = filename_string.encode()
        with open(filename, 'w') as output_file:
            csv_out = csv.writer(output_file)
            csv_out.writerow(cohort_header)
            csv_out.writerows(patient_cohort)
        print('STATUS: 0_patient_cohort.csv created.')


        ##### 2) export chart_events data for each icu_stay #####
        # Get icu_stay_ids from patient_cohort
        icu_stay_ids: list = []
        icu_stay_ids_set: set = set()
        for entry in patient_cohort:
            icu_stay_ids.append(entry[1])
            icu_stay_ids_set.add(entry[1])
        # icu_stay_ids.sort()
        if len(icu_stay_ids) != len(icu_stay_ids_set):
            print('Warning: Duplicate icustay_ids exist. Recommended to check patient_cohort.')

        # Get chart_events for each icustay and export to .csv
        first_counter = 0
        single_header = []
        for icustay_id in icu_stay_ids[:3]:             # loop through for all ids
            print('STATUS: Executing query_single_icustay for icustay_id', str(icustay_id))
            starting_time = datetime.now()
            single_icustay: list = SQL_queries.query_single_icustay(cursor_1, icustay_id)
            # print('CHECK: Query result:', single_icustay)
            if first_counter == 0:
                ending_time = datetime.now()
                time_needed = ending_time - starting_time
                print('CHECK: Seconds needed for first Query:', round(time_needed.total_seconds()))
                print(f'CHECK: Estimated minutes for all {len(icu_stay_ids)} Queries: {round(time_needed.total_seconds() * len(icu_stay_ids) / 60)}')
                single_header: list = SQL_queries.query_header_single(cursor_1)
                first_counter += 1

            filename_string: str = f'{directory}/icustay_id_{icustay_id}.csv'
            filename = filename_string.encode()
            with open(filename, 'w') as output_file:
                csv_out = csv.writer(output_file)
                csv_out.writerow(single_header)
                csv_out.writerows(single_icustay)

        cursor_1.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)

    finally:
        if conn is not None:
            conn.close()


def create_table_all_diagnoses() -> None:               # possibly also create all later needed dictionary-tables here (like the label-measurement dict)
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
        print('Connecting to the PostgreSQL database:')
        conn = psycopg2.connect(**db_params)
        cur_1 = conn.cursor()
        print('Connection successful.')

        ##### 0) Only first time: create the necessary table 'all_diagnoses_icd' #####
        ## QUERY 0
        SQL_queries.query_create_table_all_diagnoses_icd(cur_1)

        # close the communication with the database
        cur_1.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)

    finally:
        if conn is not None:
            conn.close()
