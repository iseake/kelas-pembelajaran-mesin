import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def replaceMissingValue():
    data = pd.read_csv('shoppingData_MissingValue.csv')
    data = data.values
    x = data[:, 2:5]

    df = pd.DataFrame({
        'Age': x[:, 0],
        'Income': x[:, 1],
        'Spending Score': x[:, 2],
    })
    print("datase sebelum replace missing value")
    print(df)

    imputerMean = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imputerMean = pd.DataFrame(
        imputerMean.fit_transform(df),
        columns=df.columns
    )
    print("\ndata sesudah replace missing value")
    print(df_imputerMean)


if __name__ == '__main__':
    replaceMissingValue()
