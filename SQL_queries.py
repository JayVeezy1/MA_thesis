

def query_patient_cohort(cur_1, use_case_icd_list=None):
    if use_case_icd_list is None:
        use_case_icd_list = []

    # Option 1: function inside PGAdmin CON: too complex
    # cur_1.callproc('Function_name', parameters=use_case_icd_list)  # parameters = IN, OUT

    # Option 2: complete SQL in Python, PRO: all code in one place, CON: creating objects inside DB might be better, SQL is way too long
    # query_basic_statistics: str = 'Select.....'

    # Option 3: View inside PGAdmin PRO: less complex than function
    # TODO 2. how to deal with multiple diagnosis -> cross table?
    # TODO 1. how to filter for multiple WHERE icd_9 = use_case_icd_list (list)

    # Setup icd_filter
    query_icd_filter: str = '('
    for code in use_case_icd_list:
        query_icd_filter += str(code) + ', '
    query_icd_filter = query_icd_filter[:-2] + ')'

    # Execute Query for patient cohort
    string_test = ''
    for element in query_icd_filter:
        string_test += element
    string_test = string_test[1:-1]

    query_patient_cohort_string: str = f'SELECT * FROM mimiciii.patient_cohort_with_icd where patient_cohort_with_icd.icd9_code = \'{string_test}\';'

    cur_1.execute(query_patient_cohort_string)
    print('query_basic_statistics executed.')

    return None


def get_header_patient_cohort(cur_1) -> list:
    # 'SELECT * FROM information_schema.columns WHERE table_schema = 'mimiciii' AND table_name = 'sepsis3_cohort''
    query_basic_header: str = 'SELECT column_name FROM information_schema.columns WHERE table_schema = \'mimiciii\' AND table_name = \'patient_cohort_view\''
    cur_1.execute(query_basic_header)
    header_list: list = []
    for element in cur_1.fetchall():
        header_list.append(element[0])

    return header_list