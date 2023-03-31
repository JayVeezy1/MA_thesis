import datetime

import tensorflow
from matplotlib import pyplot as plt
from pandas.core.interchange import dataframe
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler

# Packages for Deep learning:
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

# Own Functions
from step_4_classification.classification import preprocess_for_classification


# Optional idea: use GPU for Deep Learning model for more efficient and faster computation
def get_classification_report_deeplearning(use_this_function, sampling_method, selected_cohort, cohort_title,
                                           use_case_name, features_df,
                                           selected_features, selected_dependent_variable, verbose,
                                           save_to_file):
    # calculate the CM and return the corresponding ClassificationReport
    if not use_this_function:
        return None

    if verbose:
        print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))  # output = 0

    # Create basic training_test_split
    x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic = split_classification_data_DL(
        selected_cohort, cohort_title, features_df, selected_features,
        selected_dependent_variable, sampling_method, verbose)

    # define the keras model
    # Optional idea: use tf.keras.layers.GRU nodes for Deep Learning model instead of Sequential() model
    model = Sequential()                # basic keras model for deep learning network
    model.add(Dense(16, input_shape=(None, len(x_train_final.columns)), activation='relu'))
    model.add(Dense(12, activation='relu'))         # amount and setup of layers can be changed
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # alternative optimizer: 'sgd'
    # alternative use recall as metric: metrics=[tf.keras.metrics.Recall(thresholds=0)]

    # fit the keras model on the dataset + predict values y_pred
    history = model.fit(x=x_train_final, y=y_train_final, epochs=150, batch_size=32)
    y_pred = model.predict(x=x_test_final, batch_size=128).round()   # round() needed to get from sigmoid probability to class value 0 or 1

    # Get complete classification_report
    report = classification_report(y_true=y_test_final, y_pred=y_pred)

    # Get recall value
    recall_object = tensorflow.keras.metrics.Recall()
    recall_object.update_state(y_true=y_test_final, y_pred=y_pred)
    recall_value = recall_object.result().numpy()

    if verbose:
        print(f'\n CHECK: Classification Report for deeplearning on {cohort_title}, {sampling_title}:')
        print(report)
        print(f'\n CHECK: recall_object:')
        print(recall_value)

    if save_to_file:
        # Save DL classification_report
        current_time = datetime.datetime.now().strftime("%d%m%Y_%H_%M_%S")
        report_filename_string: str = f'./output/{use_case_name}/classification_deeplearning/REPORT_deeplearning_{cohort_title}_{sampling_title}_{current_time}.csv'
        report_filename = report_filename_string.encode()
        with open(report_filename, 'w', newline='') as output_file:
            output_file.write(report)
            output_file.close()
            print(f'STATUS: deeplearning classification report was saved to {report_filename}')

        # Save model configurations plot
        loss_and_metrics = model.evaluate(x=x_test_final, y=y_test_final, batch_size=128)
        fig, ax = plt.subplots()
        loss_color = '#B00000'
        ax.plot(history.history['loss'], color=loss_color)      # , marker=".")
        ax.set_xlabel('epoch', fontsize=12)
        ax.set_ylabel('loss', color=loss_color, fontsize=12)
        ax2 = ax.twinx()  # twin object for second y-axis
        accuracy_color = '#0000B0'
        ax2.plot(history.history['accuracy'], color=accuracy_color)     # , marker=".")
        ax2.set_ylabel('accuracy', color=accuracy_color, fontsize=12)
        plt.title(f'Loss({round(loss_and_metrics[0], 2)}) and Accuracy({round(loss_and_metrics[1], 2)}) for DL Model on {cohort_title}', wrap=True)
        plt.savefig(f'./output/{use_case_name}/classification_deeplearning/MODEL_deeplearning_{cohort_title}_{sampling_title}_{current_time}.png',
                    bbox_inches='tight',
                    dpi=600)
        plt.show()
        plt.close()

    return report


def split_classification_data_DL(selected_cohort: dataframe, cohort_title: str, features_df: dataframe,
                                 selected_features: list, selected_dependent_variable: str, sampling_method: str,
                                 verbose: True):
    # Classification/Prediction on avg_patient_cohort
    # Cleanup & filtering
    avg_df = preprocess_for_classification(selected_cohort=selected_cohort,
                                           features_df=features_df,
                                           selected_features=selected_features,
                                           selected_dependent_variable=selected_dependent_variable)
    death_df = avg_df[selected_dependent_variable]  # death_label as its own df y_data
    avg_df_filtered = avg_df.drop([selected_dependent_variable], axis=1)  # death_label not inside x_data

    # Split training/test-data
    x_train_basic, x_test_basic, y_train_basic, y_test_basic = train_test_split(avg_df_filtered, death_df,
                                                                                test_size=0.2, random_state=1321)

    # If selected, get over-/under-sampled data
    try:
        x_train_final, x_test_final, y_train_final, y_test_final, sampling_title = get_sampled_data_DL(sampling_method,
                                                                                                       x_train_basic,
                                                                                                       x_test_basic,
                                                                                                       y_train_basic,
                                                                                                       y_test_basic,
                                                                                                       cohort_title,
                                                                                                       verbose)
    except TypeError as e:
        return None

    return x_train_final, x_test_final, y_train_final, y_test_final, sampling_title, x_test_basic, y_test_basic


def get_sampled_data_DL(sampling_method, basic_x_train, basic_x_test, basic_y_train, basic_y_test, cohort_title,
                        verbose: True):
    if sampling_method == 'no_sampling':  # no over/undersampling
        x_train_final = basic_x_train
        y_train_final = basic_y_train
        x_test_final = basic_x_test
        y_test_final = basic_y_test
        sampling_title = 'no_sampling'
    elif sampling_method == 'oversampling':  # Oversampling: SMOTE
        if verbose:
            print(f'STATUS: Implementing SMOTE oversampling for {cohort_title}.')

        if basic_y_train.sum() < 5:
            if verbose:
                print(
                    f'WARNING: SMOTE can not be implemented for {cohort_title} because not enough death-cases for train-test-split. '
                    f'At least 5 cases needed. '
                    f'RandomOversampling is implemented instead.')
            ros = RandomOverSampler(random_state=1321)
            x_res, y_res = ros.fit_resample(basic_x_train, basic_y_train)
            new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_res, y_res,
                                                                                test_size=0.2,
                                                                                random_state=1321)
            x_train_final = new_x_train
            y_train_final = new_y_train
            x_test_final = new_x_test
            y_test_final = new_y_test
            sampling_title = 'random_oversampling'
        else:
            smote = SMOTE(random_state=1321)
            smote_x_resampled, smote_y_resampled = smote.fit_resample(basic_x_train, basic_y_train)
            new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(smote_x_resampled, smote_y_resampled,
                                                                                test_size=0.2,
                                                                                random_state=1321)
            x_train_final = new_x_train
            y_train_final = new_y_train
            x_test_final = new_x_test
            y_test_final = new_y_test
            sampling_title = 'oversampling_smote'
    else:
        # no undersampling for DL -> would need a clf here to optimize/choose the NearMiss version
        x_train_final = None
        y_train_final = None
        x_test_final = None
        y_test_final = None
        sampling_title = 'unknown_method'

    return x_train_final, x_test_final, y_train_final, y_test_final, sampling_title
