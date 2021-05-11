import pandas as pd
from imblearn.over_sampling import SMOTE

def load_dataset():
    X = pd.read_csv('datasets/spambase.csv')

    # Separates the attributes and the label in different variables
    Y = X['label'].to_numpy()
    X = X.drop('label', axis=1)  # axis = 1 -> removes the column

    return X, Y


def normalize(X):
    df_final = []

    # iterates through each column
    for (feature, data) in X.iteritems():
        column = []
        maxVal = data.max()
        minVal = data.min()

        diff = maxVal - minVal
        # iterates through each value in a column
        for i in data:
            newValue = (i - minVal) / diff
            column.append(newValue)
        df_final.append(column)

    # Transpose matrix because columns were being saved as rows
    X_new = pd.DataFrame(df_final).T
    # Adds the corresponding label to each column
    X_new.columns = X.columns

    return X_new


def main():
    X, Y = load_dataset()

    print('----------Original Dataset----------')
    print(X)

    X_balanced, Y_balanced = SMOTE(random_state=787070).fit_resample(X, Y)

    print()
    print('----------Balanced Dataset----------')
    print(X_balanced)

    X_normal = normalize(X_balanced)

    print()
    print('----------Normalized Dataset----------')
    print(X_normal)

    # Exploratory Analysis
    print('----------Exploratory Analysis----------')
    print(X_normal.describe(()))


if __name__ == '__main__':
    main()

