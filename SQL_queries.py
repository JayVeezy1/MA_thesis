# Option 1: complete SQL in Python,
# PRO: all code in one place, CON: creating objects inside DB is better, query is way too long
# Option 2: Use a function in PGAdmin CON: works, but view is easier -> choose option 3
# cur_1.callproc('Function_name', parameters=use_case_icd_list)
# Option 3: Call a View that was created in PGAdmin
def query_patient_cohort(cur_1, use_case_icd_list=None):
    if use_case_icd_list is None:
        use_case_icd_list = []

    # TODO check if filter with this long string works
    # TODO remove duplicate hadm_id rows (keep where seq_num is minimum)
    # TODO add 'all_icd9' column to query 1

    # Setup query_icd_filter as string
    query_with_icd_filter: str = f'SELECT * FROM mimiciii.patient_cohort_with_icd where patient_cohort_with_icd.icd9_code = {use_case_icd_list[0]}'
    if len(use_case_icd_list) > 1:
        for icd_code in use_case_icd_list[1:]:
            query_with_icd_filter += ' OR patient_cohort_with_icd.icd9_code = ' + str(icd_code)
        query_with_icd_filter = query_with_icd_filter[:-44] + ';'

    print('TEST QUERY:', query_with_icd_filter)

    # Execute Query for patient cohort
    query_patient_cohort_string: str = f'SELECT * FROM mimiciii.patient_cohort_with_icd where patient_cohort_with_icd.icd9_code = \'{query_with_icd_filter}\';'
    cur_1.execute(query_patient_cohort_string)
    print('query_basic_statistics executed.')

    return None


def get_header_patient_cohort(cur_1) -> list:
    query_basic_header: str = 'SELECT column_name FROM information_schema.columns WHERE table_schema = \'mimiciii\' AND table_name = \'patient_cohort_with_icd\';'
    cur_1.execute(query_basic_header)
    print('query_basic_header executed.')

    header_list: list = []
    for element in cur_1.fetchall():
        header_list.append(element[0])

    return header_list


def query_create_table_all_diagnoses_icd(cur_1):
    query_create_table_all_diagnoses_icd_string: str = 'select get_all_diagnoses();'
    cur_1.execute(query_create_table_all_diagnoses_icd_string)
    print('query_create_table_all_diagnoses_icd executed.')

    return None
