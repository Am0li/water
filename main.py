import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #Spliting data into training and test sets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report #used to measure output of models
from sklearn.tree import DecisionTreeClassifier #import model: Decison tree
from sklearn import tree
import matplotlib.pyplot as plt

def clean_data(df, remove_null):
    ###Cleaning data
    # Removing Duplicates
    df = df.drop_duplicates()

    ##empty or null cells
    if remove_null: # option 1:drop rows with empty cells
        df = df.dropna()
    else: #option 2: fill cells
        for column in df.select_dtypes(include=['number']).columns:
            med = df[column].median()
            df[column] = df[column].fillna(med)
    return df


#import data from csv
df_water = pd.read_csv("jakos_wody.csv")
#locate row
#print(df_water.loc[0])
#dataframe info
#print(df_water.info())

df_clean = clean_data(df_water, True)

#split data for training and testing
X = df_clean.drop("Potability",axis=1).values
y = df_clean["Potability"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=101)

clf_tree = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=7, min_samples_leaf=30)
clf_tree.fit(X_train,y_train)

y_pred = clf_tree.predict(X_test)

print("Confusion Matrix: ",
      confusion_matrix(y_test, y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
print("Report : ", classification_report(y_test, y_pred))

plt.figure(figsize=(15, 10))
tree.plot_tree(clf_tree,feature_names=df_clean.columns[0:-1], filled=True)

cm = confusion_matrix(y_test, y_pred, labels=clf_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf_tree.classes_)
disp.plot()

plt.show()

