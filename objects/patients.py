import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class NotUniqueIDError(Exception):
    pass


class Patient:
    """
    Class containing all patients that are imported from the .csv files.

    Note: The .csv files/ids are actually based on icustays. A real patient might have had multiple icustays,
    but only their first icustay will be included in this analysis. Meaning the icustay_id will be used as 'patient_id'.
    """

    # CLASS FEATURES
    SELECTED_FEATURES: list = []                    # IMPORTANT: this must be changed manually to the features selected in step_1
    all_patients_set: set = set()


    def __init__(self, patient_id: str, patient_data: pd.DataFrame):
        self.data = patient_data
        self.features = list(self.data.columns.values)

        if patient_id not in Patient.all_patients_set:
            self.patient_id = patient_id
            Patient.all_patients_set.add(patient_id)
        else:
            raise NotUniqueIDError(f'Icustay ID {self.patient_id} already exists in all_patients_set')

        for feature in Patient.SELECTED_FEATURES:
            if feature not in self.features:
                raise ValueError(f'Data columns of patient {self.patient_id} do not match the SELECTED_FEATURES.')


    def __del__(self):
        if self.patient_id in Patient.all_patients_set:
            Patient.all_patients_set.remove(self.patient_id)
