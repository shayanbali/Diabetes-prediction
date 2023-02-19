

import pandas as pd

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(df.head())
    df.hist()
    # create a subplot of 3 x 3
    plt.subplots(3, 3, figsize=(15, 15))
    # Plot a density plot for each variable
    for idx, col in enumerate(df.columns):
        ax = plt.subplot(3, 3, idx + 1)
        ax.yaxis.set_ticklabels([])
        sns.distplot(df.loc[df.Outcome == 0][col], hist=False, axlabel=False,
                     kde_kws={'linestyle': '-',
                              'color': 'black', 'label': "No Diabetes"})
        sns.distplot(df.loc[df.Outcome == 1][col], hist=False, axlabel=False,
                     kde_kws={'linestyle': '--',
                              'color': 'black', 'label': "Diabetes"})
        ax.set_title(col)
    # Hide the 9th subplot (bottom right) since there are only 8 plots
    plt.subplot(3, 3, 9).set_visible(False)
    print(df.isnull().any())
    print(df.describe())

    df['Glucose'] = df['Glucose'].replace(0, np.nan)
    df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
    df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
    df['Insulin'] = df['Insulin'].replace(0, np.nan)
    df['BMI'] = df['BMI'].replace(0, np.nan)
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col] == 0].shape[0]
        print(col + ": " + str(missing_rows))
    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Outcome'] = df['Outcome']
    df = df_scaled
    print(df.describe().loc[['mean', 'std', 'max'], ].round(2).abs())

    X = df.loc[:, df.columns != 'Outcome']
    y = df.loc[:, 'Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    model = Sequential()
    # Add the first hidden layer
    model.add(Dense(32, activation='relu', input_dim=8))
    # Add the second hidden layer
    model.add(Dense(16, activation='relu'))
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=400)
    scores = model.evaluate(X_train, y_train)
    print("Training Accuracy: %.2f%%\n" % (scores[1] * 100))
    scores = model.evaluate(X_test, y_test)
    print("Testing Accuracy: %.2f%%\n" % (scores[1] * 100))

    # y_test_pred = model.predict_classes(X_test)
    predict_x = model.predict(X_test)
    y_test_pred = np.argmax(predict_x, axis=1) #class
    c_matrix = confusion_matrix(y_test, y_test_pred)
    ax = sns.heatmap(c_matrix, annot=True,
                     xticklabels=['No Diabetes', 'Diabetes'],
                     yticklabels=['No Diabetes', 'Diabetes'],
                     cbar=False, cmap='Blues')
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")

    y_test_pred_probs = model.predict(X_test)
    FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
    plt.plot(FPR, TPR)
    plt.plot([0, 1], [0, 1], '--', color='black')  # diagonal line
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.show()

