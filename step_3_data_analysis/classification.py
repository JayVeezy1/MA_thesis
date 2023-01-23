import datetime

import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, \
    roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss


def calculate_RF_on_cohort(avg_patient_cohort, cohort_title, selected_features, selected_dependent_variable, save_to_file):
    # Classification/Prediction with RandomForest on avg_patient_cohort

    # Preprocessing for classification
    avg_df = avg_patient_cohort.drop(columns=['icustay_id',
                                              'stroke_type']).copy()  # these features are needed for clustering, not correlations/classification
    avg_df = avg_df.drop(columns=['ethnicity', 'insurance'])        # todo: these features can only be used if numeric - any good way to include?
    selected_features_final = selected_features
    selected_features_final.remove('icustay_id')
    selected_features_final.remove('stroke_type')
    selected_features_final.remove('ethnicity')
    selected_features_final.remove('insurance')
    selected_features_final.append(selected_dependent_variable)

    available_dependent_variables: list = ['death_in_hosp', 'death_3_days', 'death_30_days', 'death_180_days', 'death_365_days']
    available_dependent_variables.remove(selected_dependent_variable)
    avg_df = avg_df.drop(columns=available_dependent_variables)

    # Implement Training
    avg_df_filtered = avg_df[selected_features_final]
    avg_df_filtered = avg_df_filtered.fillna(0)     # todo: is this fix of NaN problem correct?
    death_df = avg_df_filtered[selected_dependent_variable]         # death_label as its own df y_data
    avg_df_filtered = avg_df_filtered.drop([selected_dependent_variable], axis=1)   # death_label not inside x_data

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, bootstrap=True, random_state=1321)
    x_train, x_test, y_train, y_test = train_test_split(avg_df_filtered, death_df, test_size=0.2, random_state=1321)

    # this is the training step, .prediction will be inside the Classification Report
    clf.fit(X=x_train, y=y_train)

    # Classification Report
    display_confusion_matrix(clf, x_test, y_test, cohort_title=cohort_title, save_to_file=save_to_file)

    # ROC/AUC Curve
    display_roc_auc_curve(clf, x_test, y_test, cohort_title=cohort_title)

    # todo: add over/undersampling
    """
    # Oversampling: SMOTE
    smote = SMOTE(random_state=1337)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_train_smote, y_train_smote, test_size=0.2,
                                                                        random_state=1321)
    clf.train(x_train_smote, y_train_smote)
    print("Classification Report for SMOTE oversampling")
    display_confusion_matrix(clf, new_x_test, new_y_test, plotting=True, version="SMOTE oversampling", set=set)
    display_roc_auc_curve(clf, new_x_test, new_y_test, version="SMOTE oversampling", plotting=True, set=set)

    # Undersampling: NearMiss
    versions = [1, 2, 3]
    for version in versions:
        near_miss = NearMiss(version=version)
        new_x, new_y = near_miss.fit_resample(x_test, y_test)
        new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_x, new_y, test_size=0.2,
                                                                            random_state=1321)
        clf.train(x_data=new_x_train, y_data=new_y_train)
        print("Classification Report for NearMiss Version:", version)
        display_confusion_matrix(clf, new_x_test, new_y_test, plotting=True, version="NearMiss"+str(version), set=set)
        display_roc_auc_curve(clf, new_x_test, new_y_test, version="NearMiss"+str(version), plotting=True, set=set)
    """
    return None



def display_confusion_matrix(clf, x_test, y_test, cohort_title, save_to_file):
    cm: ndarray = confusion_matrix(y_test, clf.predict(x_test))
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Death", "Death"])
    disp.plot(ax=ax)            # todo: any option to make "true" case at top and "false (no_death)" to bottom row?
    ax.set_title(f"RandomForest Classification for {cohort_title}")
    if save_to_file:
        plt.savefig(f'./output/classification/random_forest_{cohort_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
    plt.show()

    # Print Confusion Matrix and Classification Report
    print(f'Confusion Matrix for {cohort_title}:')
    cm_df = pd.DataFrame({
        "predicts_true": {"is_true": cm[1][1], "is_false": cm[0][1]},
        "predicts_false": {"is_true": cm[1][0], "is_false": cm[0][0]}
    })
    print(cm_df)

    print(f'\n Classification Report for {cohort_title}:')
    report = classification_report(y_test, clf.predict(x_test))
    print(report)


def display_roc_auc_curve(clf, x_test, y_test, cohort_title):
    # todo: this is not correct yet!
    roc_auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])

    disp = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=clf.predict(x_test), name=f'Random Forest for {cohort_title}')

    plt.title(f"RandomForest for {cohort_title} AUC: {round(roc_auc, 4)}")

    plt.show()
