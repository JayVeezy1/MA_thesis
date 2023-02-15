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
from xgboost import XGBClassifier


def preprocess_for_classification(avg_cohort, features_df, selected_features, selected_dependent_variable):
    # Preprocessing for Clustering: Remove the not selected prediction_variables and icustay_id
    prediction_variables = features_df['feature_name'].loc[
        features_df['potential_for_analysis'] == 'prediction_variable'].to_list()
    for feature in prediction_variables:
        try:
            selected_features.remove(feature)
        except ValueError as e:
            pass
    selected_features.append(selected_dependent_variable)  # keeping selected_dependent_variable for clustering?
    try:
        selected_features.remove('icustay_id')
    except ValueError as e:
        pass

    return avg_cohort[selected_features].fillna(0)


def get_sampled_data(clf, sampling_method, basic_x_train, basic_x_test, basic_y_train, basic_y_test, cohort_title):
    if sampling_method == 'no_sampling':  # no over/undersampling
        x_train_final = basic_x_train
        y_train_final = basic_y_train
        x_test_final = basic_x_test
        y_test_final = basic_y_test
        sampling_title = 'no_sampling'

    elif sampling_method == 'oversampling':  # Oversampling: SMOTE
        print(f'STATUS: Implementing SMOTE oversampling for {cohort_title}.')
        smote = SMOTE(random_state=1321)
        smote_x_data, smote_y_data = smote.fit_resample(basic_x_train, basic_y_train)
        new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(smote_x_data, smote_y_data, test_size=0.2,
                                                                            random_state=1321)
        x_train_final = new_x_train
        y_train_final = new_y_train
        x_test_final = new_x_test
        y_test_final = new_y_test
        sampling_title = 'oversampling_smote'

    elif sampling_method == 'undersampling':  # Undersampling: NearMiss
        # Undersampling: NearMiss
        x_train_final = basic_x_train
        y_train_final = basic_y_train
        x_test_final = basic_x_test
        y_test_final = basic_y_test
        sampling_title = f'undersampling_nearmiss_FAILED'
        versions = [1, 2, 3]
        best_recall = 0
        # try all 3 nearmiss versions
        for version in versions:
            print(f'STATUS: Implementing NearMiss {version} oversampling for {cohort_title}.')
            near_miss = NearMiss(version=version)
            near_miss_x_data, near_miss_y_data = near_miss.fit_resample(basic_x_train, basic_y_train)
            new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(near_miss_x_data, near_miss_y_data,
                                                                                test_size=0.2, random_state=1321)
            clf.fit(X=new_x_train, y=new_y_train)

            # Choose best recall of all nearmiss versions
            temp_cm = confusion_matrix(new_y_test, clf.predict(new_x_test))
            temp_recall = temp_cm[1][1] / (
                    temp_cm[1][1] + temp_cm[1][0])  # Recall = TruePositives / (TruePositives + FalseNegatives)
            if temp_recall >= best_recall:
                best_recall = temp_recall
                x_train_final = new_x_train
                y_train_final = new_y_train
                x_test_final = new_x_test
                y_test_final = new_y_test
                sampling_title = f'undersampling_nearmiss_v{version}'
    else:
        print(
            f'ERROR: sampling_method {sampling_method} not valid. Choose from options: "no_sampling" or "oversampling" or "undersampling"')
        return None

    return x_train_final, x_test_final, y_train_final, y_test_final, sampling_title


def calculate_classification_on_cohort(avg_cohort, cohort_title, use_case_name, features_df, selected_features,
                                       selected_dependent_variable, save_to_file, classification_method: str,
                                       sampling_method: str):
    # Classification/Prediction with RandomForest on avg_patient_cohort
    # Cleanup & filtering
    avg_df = preprocess_for_classification(avg_cohort, features_df, selected_features, selected_dependent_variable)
    death_df = avg_df[selected_dependent_variable]  # death_label as its own df y_data
    avg_df_filtered = avg_df.drop([selected_dependent_variable], axis=1)  # death_label not inside x_data

    # Create basic training_test_split
    if classification_method == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, bootstrap=True,
                                     random_state=1321)  # todo: why boostrap=True ? == CrossValidation?
    elif classification_method == 'XGBoost':
        clf = XGBClassifier()
    else:
        print(
            f'ERROR: classification_method "{classification_method}" not valid. Choose from options: "RandomForest" or "XGBoost".')
        return None

    # Split training/test-data
    basic_x_train, basic_x_test, basic_y_train, basic_y_test = train_test_split(avg_df_filtered, death_df,
                                                                                test_size=0.2, random_state=1321)
    # If selected, get over-/under-sampled data
    try:
        x_train_final, x_test_final, y_train_final, y_test_final, sampling_title = get_sampled_data(clf,
                                                                                                    sampling_method,
                                                                                                    basic_x_train,
                                                                                                    basic_x_test,
                                                                                                    basic_y_train,
                                                                                                    basic_y_test,
                                                                                                    cohort_title)
    except TypeError as e:
        return None

    # this is the training step, prediction will be inside the Classification Report
    clf.fit(X=x_train_final, y=y_train_final)
    # Classification Report
    display_confusion_matrix(clf, x_test_final, y_test_final, cohort_title=cohort_title, use_case_name=use_case_name,
                             sampling_title=sampling_title, save_to_file=save_to_file)
    # ROC/AUC Curve
    display_roc_auc_curve(clf, x_test_final, y_test_final, cohort_title=cohort_title, use_case_name=use_case_name,
                          sampling_title=sampling_title, save_to_file=save_to_file)

    return None


def display_confusion_matrix(clf, x_test, y_test, cohort_title, use_case_name, sampling_title, save_to_file):
    cm: ndarray = confusion_matrix(y_test, clf.predict(x_test))
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Death", "Death"])
    disp.plot(ax=ax)  # todo: any option to make "true" case at top and "false (no_death)" to bottom row?
    ax.set_title(f"RandomForest Classification for {cohort_title}, {sampling_title}")

    current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
    if save_to_file:
        plt.savefig(f'./output/{use_case_name}/classification/RF_{cohort_title}_{sampling_title}_{current_time}.png')
    plt.show()

    # Print Confusion Matrix and Classification Report
    print(f'Confusion Matrix for {cohort_title}, {sampling_title}:')
    cm_df = pd.DataFrame({
        "predicts_true": {"is_true": cm[1][1], "is_false": cm[0][1]},
        "predicts_false": {"is_true": cm[1][0], "is_false": cm[0][0]}
    })
    print(cm_df)

    print(f'\n Classification Report for {cohort_title}, {sampling_title}:')
    report = classification_report(y_test, clf.predict(x_test))
    print(report)

    if save_to_file:
        cm_filename_string: str = f'./output/{use_case_name}/classification/RF_{cohort_title}_{sampling_title}_confusion_matrix_{current_time}.csv'
        cm_filename = cm_filename_string.encode()
        with open(cm_filename, 'w', newline='') as output_file:
            cm_df.to_csv(output_file)
            print(f'STATUS: cm_df was saved to {cm_filename}')

        report_filename_string: str = f'./output/{use_case_name}/classification/RF_{cohort_title}_{sampling_title}_report_{current_time}.csv'
        report_filename = report_filename_string.encode()
        with open(report_filename, 'w', newline='') as output_file:
            output_file.write(report)
            output_file.close()


# todo: this is not correct yet!
def display_roc_auc_curve(clf, x_test, y_test, cohort_title, use_case_name, sampling_title, save_to_file):
    roc_auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])

    disp = RocCurveDisplay.from_predictions(y_true=y_test, y_pred=clf.predict(x_test),
                                            name=f'Random Forest for {cohort_title}, {sampling_title}')

    plt.title(f"RandomForest for {cohort_title} AUC: {round(roc_auc, 4)}, {sampling_title}")

    if save_to_file:
        # save disp once its correct
        # plt.savefig(f'./output/{use_case_name}/classification/random_forest_{cohort_title}_{sampling_title}_{datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")}.png')
        pass

    plt.show()
