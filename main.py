import mimic_to_csv

####### MAIN #######
if __name__ == '__main__':
    mimic_to_csv.export_unique_adm_to_csv(use_case_icd_list=[42731],
                                          use_case_name='testing')
