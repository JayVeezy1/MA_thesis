import csv

import psycopg2

import db_setup

####### MAIN #######

# Connect with PostGre MIMIC-III Database
db_params = db_setup.config()
conn = None
try:
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database:')
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()

    # execute a statement
    query_1 = 'SELECT row_id FROM mimiciii.patients'  # TODO this needs to be unique key: hadm&subject_id
    cur.execute(query_1)
    print('Main Query 1 executed.')

    # export data
    cur_2 = conn.cursor()
    for selected_row_id in range(1, cur.rowcount):
        query_2 = f'SELECT * FROM mimiciii.patients WHERE mimiciii.patients.row_id = {selected_row_id}'
        cur_2.execute(query_2)

        # TODO get column names: query_3 = 'select * from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME=<table name>'
        # this probably needs to be a real view that is created in query_2

        # Export patient_entry to csv
        patient_entry: list = cur_2.fetchall()  # this only works for 1 row currently

        final_list = []
        # each row per patient is one string in the final list
        for row in patient_entry:
            final_list.append(row)

        # remove unnecessary brackets at beginning and end of row
        final_list[0] = str(final_list[0])[1:]
        final_list[0] = str(final_list[0])[:-1]
        data_list = final_list[0].split(',')

        filename_string: str = f'C:/Users/Jakob/Documents/Studium/Master_Frankfurt/Masterarbeit/MIMIC_III/my_queries/export/patient_{selected_row_id}.csv'
        filename = filename_string.encode()
        with open(filename, 'w') as output_file:
            csv_out = csv.writer(output_file)
            csv_out.writerow(['row_id', 'subject_id', 'gender', 'dob', 'dod', 'dod_hosp', 'dod_ssn', 'expire_flag'])
            csv_out.writerow(data_list)

    # close the communication with the PostgreSQL
    cur.close()
    cur_2.close()
except (Exception, psycopg2.DatabaseError) as error:
    print('Error occurred:', error)

finally:
    if conn is not None:
        conn.close()
