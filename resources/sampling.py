from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import AllKNN
import pandas as pd

def over_sampling(xTrain, yTrain, model='ADASYN', neighbors=200):
    """
    It generate synthetic sampling for the minority class using the model specificed. Always it has
    to be applied to the training set.
    :param xTrain: X training set.
    :param yTrain: Y training set.
    :param model: 'ADASYN' or 'SMOTE'
    :param neighbors: number of nearest neighbours to used to construct synthetic samples.
    :return: xTrain and yTrain oversampled
    """

    xTrainNames = xTrain.columns.values.tolist()
    yTrainNames = ['target']

    if model == 'ADASYN':
        model = ADASYN(random_state=42, ratio='minority', n_neighbors=neighbors)

    if model == 'SMOTE':
        model = SMOTE(random_state=42, ratio='minority', k_neighbors=neighbors, m_neighbors='svm')

    xTrain, yTrain = model.fit_sample(xTrain, yTrain)

    xTrain = pd.DataFrame(xTrain, columns=[xTrainNames])
    yTrain = pd.DataFrame(yTrain, columns=[yTrainNames])

    return xTrain, yTrain


def under_sampling(xTrain, yTrain, neighbors=200):
    """
    It reduces the sample size for the majority class using the model specificed. Always it has
    to be applied to the training set.
    :param xTrain: X training set.
    :param yTrain: Y training set.
    :param neighbors: size of the neighbourhood to consider to compute the
        average distance to the minority point samples
    :return: xTrain and yTrain oversampled
    """

    xTrainNames = xTrain.columns.values.tolist()
    yTrainNames = yTrain.columns.values.tolist()

    model = AllKNN(random_state=42, ratio='majority', n_neighbors=neighbors)

    xTrain, yTrain = model.fit_sample(xTrain, yTrain)

    xTrain = pd.DataFrame(xTrain, columns=[xTrainNames])
    yTrain = pd.DataFrame(yTrain, columns=[yTrainNames])

    return xTrain, yTrain