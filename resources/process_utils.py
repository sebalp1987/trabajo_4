import pandas as pd
import STRING
from sklearn.model_selection import train_test_split
import datetime


def output_normal_anormal_new(df: pd.DataFrame, output_file=True, key_var_split='target'):
    """
    It split the dataframe into three dataframes (normal, anormal) based on target = (0,1)
    . Also, if 'output_file' = True, it creates a new version of the final table.
    :param df: Original dataframe
    :param output_file: Boolean if it is necessary an output file
    :param key_var_split: The name of the key variable used to split.
    :return: Two dataframes based on normally anormally.
    """

    # Split dataframe
    df[key_var_split] = df[key_var_split].map(int)

    anomaly = df[df[key_var_split] == 1]
    normal = df[df[key_var_split] == 0]

    print('anomaly shape ', anomaly.shape)
    print('normal shape ', normal.shape)

    if output_file:
        anomaly.describe().to_csv(STRING.anormal_describe, index=False, sep=';')
        normal.describe().to_csv(STRING.normal_describe, index=False, sep=';')
        print(anomaly.describe())
        anomaly.to_csv(STRING.anormal_file, sep=';', index=False)
        normal.to_csv(STRING.normal_file, sep=';', index=False)

    return normal, anomaly


def training_test_valid(df: pd.DataFrame, output_file=True, key_var_split='target'):
    """
    Separate between training, test and valid using the next proportions:
    Training 70%
    Test 15%
    Valid 15%
    Here, we include in the Training Set either normal cases and anormal cases using the proportions
    derivated from the original distribution.
    Then we split between Test and Valid using the same original proportions.
    """
    normal, anomaly = output_normal_anormal_new(df, False, key_var_split)

    normal = normal.reset_index(drop=True)
    anomaly = anomaly.reset_index(drop=True)

    normal_train, normal_test, _, _ = train_test_split(normal, normal, test_size=.3, random_state=10)
    anormal_train, anormal_test, _, _ = train_test_split(anomaly, anomaly, test_size=.3, random_state=10)
    normal_valid, normal_test, _, _ = train_test_split(normal_test, normal_test, test_size=.5, random_state=10)
    anormal_valid, anormal_test, _, _ = train_test_split(anormal_test, anormal_test, test_size=.5, random_state=10)

    train = normal_train.append(anormal_train).sample(frac=1).reset_index(drop=True)
    valid = normal_valid.append(anormal_valid).sample(frac=1).reset_index(drop=True)
    test = normal_test.append(anormal_test).sample(frac=1).reset_index(drop=True)

    print('Train shape: ', train.shape)
    print('Proportion os anomaly in training set: %.2f\n', train['target'].mean())
    print('Valid shape: ', valid.shape)
    print('Proportion os anomaly in validation set: %.2f\n', valid['target'].mean())
    print('Test shape:, ', test.shape)
    print('Proportion os anomaly in test set: %.2f\n', test['target'].mean())

    if output_file:
        train.to_csv(STRING.train, sep=';', index=False)
        test.to_csv(STRING.test, sep=';', index=False)
        valid.to_csv(STRING.valid, sep=';', index=False)

    return train, valid, test


def calculate_age(birthdate, sep=''):
    birthdate = datetime.datetime.strptime(birthdate, '%Y' + sep + '%m' + sep + '%d')
    today = datetime.date.today()
    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
