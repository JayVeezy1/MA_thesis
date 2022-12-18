# Brainstorming:
# Option 1: complete SQL in Python,
# PRO: all code in one place, CON: creating objects inside DB is better, query is way too long
# Option 2: Use a function in PGAdmin CON: works, but view is easier -> choose option 3
# cur_1.callproc('Function_name', parameters=use_case_icd_list)
# Option 3: Call a View that was created in PGAdmin

# Short explanation of SQL Querying steps:
# 1. First call is of function 'get_filtered_patient_cohort' with user-input icd-codes as filter parameter
# 2. this function uses the view 'patient_cohort_with_icd' where the 'patient_cohort_view' is joined with the diagnoses (with the extra field 'all_icd_codes') on the hadm_id field
# 3. the patient_cohort_view is created previously based on code by previous research (reference in the code) as they used very reasonable filtering steps.


def query_patient_cohort(cur_1, use_case_icd_list=None):
    if use_case_icd_list is None:
        use_case_icd_list = []

    # FILTERING: With array-string = '{icd, icd, icd}'
    icd_array_string: str = '\'{'
    for icd_code in use_case_icd_list:
        icd_array_string += str(icd_code) + ','
    if len(icd_array_string) > 2:
        icd_array_string = icd_array_string[:-1] + '}\''
    else:                                                                   # if no icd9 code was selected
        icd_array_string = icd_array_string + '}\''

    query_check_count: str = f'SELECT COUNT(*) FROM get_filtered_patient_cohort({icd_array_string})'
    query_with_icd_filter: str = f'SELECT * FROM get_filtered_patient_cohort({icd_array_string})'
    # print('CHECK: ICD9 Filter QUERY:', query_with_icd_filter)

    # Execute Query for patient cohort
    cur_1.execute(query_check_count)
    count = cur_1.fetchall()[0][0]
    cur_1.execute(query_with_icd_filter)
    print('STATUS: query_basic_statistics executed.')
    print('CHECK:  Count() of entries for the selected ICD9 codes=', count)

    return None


def query_create_table_all_diagnoses_icd(cur_1):
    query_create_table_all_diagnoses_icd_string: str = 'select get_all_diagnoses();'
    cur_1.execute(query_create_table_all_diagnoses_icd_string)
    print('query_create_table_all_diagnoses_icd executed.')

    return None


""" Not used anymore, simply hardcoded. Maybe not ideal but it works for now.
def get_header_patient_cohort(cur_1) -> list:
    query_basic_header: str = 'SELECT column_name FROM information_schema.columns WHERE table_schema = \'mimiciii\' AND table_name = \'patient_cohort_with_icd\';'
    cur_1.execute(query_basic_header)
    print('query_basic_header executed.')

    header_list: list = []
    for element in cur_1.fetchall():
        header_list.append(element[0])

    return header_list
"""