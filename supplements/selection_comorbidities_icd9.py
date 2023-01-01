# This file is used to categorize the necessary icd9 codes for the comorbidities

# comorbidities (icd9 filtering):
# https://cumming.ucalgary.ca/sites/default/files/teams/30/resources/Pub_ICD10%E2%80%93ICD9_ElixhauserCharlson_coding_Coding_Algorithm_for_Defining_Comorbidities_in_ICD9-CM_and%20ICD10_Admin_Data.pdf

# diabetes: 250.0 - 250.3, 250.4-250.6, 250.7, 250.9
diabetes_list = [25000, 25010, 25020, 25030, 25040, 25050, 25060, 25070, 25090]

# cancer: 140.x - 172.x, 174.x - 195.8, 200.x - 208.x, 196.x-199.1, V10.x

# hypertension: 401.1, 401.9, 402.10, 402.90, 404.10, 404.90, 405.1, 405.9
hypertension_list = [4011, 4019, 40210, 40290, 40410, 40490, 4051, 4059]


# obesity: 278.0

# alcohol/drug abuse: 291.x, 303.9, 305.0, V113, 292.0, 292.x, 304.0, 305.2-305.9
