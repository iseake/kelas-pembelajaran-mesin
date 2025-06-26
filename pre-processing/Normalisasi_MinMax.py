import pandas as pd
from sklearn import preprocessing

def perhitunganMinMax():
    data = pd.read_csv('shoppingData_MiniMax.csv')
    data = data.values
    # print(data)

    x = data[:, 2:5]
    # print(x)
    df = pd.DataFrame({
        'Age': x[:, 0],
        'Income': x[:, 1],
        'Spending Score': x[:, 2],
    })
    print('Data sebelum normalisasi')
    print(df)

    minMax = preprocessing.MinMaxScaler(feature_range=(0, 1))
    df_minMax = pd.DataFrame(
        minMax.fit_transform(df),
        columns=df.columns
    )

    print('\ndata sesudah normalisasi')
    print(df_minMax)

if __name__ == '__main__':
    perhitunganMinMax()