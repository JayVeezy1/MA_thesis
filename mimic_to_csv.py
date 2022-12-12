import csv
import os

import psycopg2
from db_setup import config


def export_unique_adm_to_csv(use_case_icd_list=None, use_case_name=None) -> None:
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
        print('Connecting to the PostgreSQL database:')
        conn = psycopg2.connect(**db_params)
        cur_1 = conn.cursor()
        print('Connection successful.')

        ##### 1) execute the query_basic_statistics procedure #####
        ## QUERY 1
        query_basic_statistics: str = 'SELECT row_id, subject_id FROM mimiciii.patients'
        # TODO this needs to be unique key: hadm&subject_id not just row_id
        # TODO also needs to WHERE icd_9 = use_case_icd_list (list)
        cur_1.execute(query_basic_statistics)
        basic_statistics: list = cur_1.fetchall()
        print('query_basic_statistics executed.')

        header_basic_statistics = ['row_id', 'subject_id']  # todo: make automatic (see header_single_adm)

        # create new directory for this use-case if it does not exist yet
        directory: str = f'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/exports/{use_case_name}'
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass

        # export to csv
        filename_string: str = f'{directory}/0_basic_statistics.csv'
        filename = filename_string.encode()
        with open(filename, 'w') as output_file:
            csv_out = csv.writer(output_file)
            csv_out.writerow(header_basic_statistics)
            csv_out.writerows(basic_statistics)

        ##### 2) export data for each single_admission #####
        cur_2 = conn.cursor()
        # TODO header_single_adm = 'select * from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME=<table name>', this probably needs to be based on the view-object that is used in query_2
        header_single_adm = ['row_id', 'subject_id', 'gender', 'dob', 'dod', 'dod_hosp', 'dod_ssn', 'expire_flag']

        for selected_unique_admission in range(1, 10):    # cur_1.rowcount): TODO: select row_id from basic_statistics instead of rowcount! or maybe even hadm&subject_id
            ## QUERY 2
            query_single_admission: str = f'SELECT * FROM mimiciii.patients WHERE mimiciii.patients.row_id = {selected_unique_admission}'
            cur_2.execute(query_single_admission)
            single_admission: list = cur_2.fetchall()
            print('query_single_admission executed for: ', selected_unique_admission)

            # each row per single_admission is one string in the final list -> TODO: check if this works for time-series, probably better to do it with writerows like in basic_statistics
            final_list: list = []
            for row in single_admission:
                final_list.append(row)
            final_list[0] = str(final_list[0])[1:]      # bei mehreren rows das hier auch einschieben und für jedes [row] machen?
            final_list[0] = str(final_list[0])[:-1]
            data_list = final_list[0].split(',')

            # export to csv
            filename_string: str = f'{directory}/patient_{selected_unique_admission}.csv'
            filename = filename_string.encode()
            with open(filename, 'w') as output_file:
                csv_out = csv.writer(output_file)
                csv_out.writerow(header_single_adm)
                csv_out.writerow(data_list)

        # close the communication with the database
        cur_1.close()
        cur_2.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print('Error occurred:', error)

    finally:
        if conn is not None:
            conn.close()