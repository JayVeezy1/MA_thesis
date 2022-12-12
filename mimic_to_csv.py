import csv
import os

import psycopg2
from db_setup import config


def export_unique_adm_to_csv(use_case_icd_list: list, use_case_name: str = 'no_use_case_name') -> None:
    """
    This function exports a .csv file per unique-admission (each patient once).
    The files are saved inside /export/use_case_name/
    The function does not return anything. Analysis of the selection will only be conducted based on the .csv files
    :param use_case_icd_list: a list of all icd_9 codes that were selected for this use-case
    :param use_case_name: the title for this use-case e.g. stroke, heart_failure, sepsis
    :return: None
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

        ##### execute the select procedure #####
        query_unique_ids: str = 'SELECT row_id FROM mimiciii.patients'
        # TODO this needs to be unique key: hadm&subject_id not just row_id
        # TODO also needs to WHERE icd_9 = use_case_icd_list (list)
        cur_1.execute(query_unique_ids)
        print('query_unique_id executed.')

        # create new directory for this use-case if it does not exist yet
        directory: str = f'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/exports/{use_case_name}'
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        # export data per unique key
        cur_2 = conn.cursor()
        for selected_row_id in range(1, 10):    # cur_1.rowcount): TODO: change back to rowcount
            query_single_admission: str = f'SELECT * FROM mimiciii.patients WHERE mimiciii.patients.row_id = {selected_row_id}'
            cur_2.execute(query_single_admission)
            single_admission: list = cur_2.fetchall()
            print('query_single_admission executed for: ', selected_row_id)

            # TODO get column names: query_3 = 'select * from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME=<table name>', this probably needs to be a real view that is created in query_2

            # each row per single_admission is one string in the final list -> TODO: check if this works for time-series
            final_list: list = []
            for row in single_admission:
                final_list.append(row)
            final_list[0] = str(final_list[0])[1:]      # bei mehreren rows das hier auch einschieben und für jedes [row] machen?
            final_list[0] = str(final_list[0])[:-1]
            data_list = final_list[0].split(',')

            # export to csv
            filename_string: str = f'{directory}/patient_{selected_row_id}.csv'
            filename = filename_string.encode()
            with open(filename, 'w') as output_file:
                csv_out = csv.writer(output_file)
                csv_out.writerow(['row_id', 'subject_id', 'gender', 'dob', 'dod', 'dod_hosp', 'dod_ssn', 'expire_flag'])
                csv_out.writerow(data_list)

        # close the communication with the database
        cur_1.close()
        cur_2.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)

    finally:
        if conn is not None:
            conn.close()
