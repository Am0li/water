import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV #Spliting data into training and test sets
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

#Parameters grid for tuning
params = {
    'max_depth': [2, 3, 5, 10, 20, 30],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}

#parameter tuning
clf_tree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=clf_tree,param_grid=params, cv=5, n_jobs=-1)
grid_search.fit(X_train,y_train)
best_tree = grid_search.best_estimator_
y_pred = best_tree.predict(X_test)

print("Best Params:",grid_search.best_params_)
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred)*100)
print("Report : ", classification_report(y_test, y_pred))

plt.figure(figsize=(15, 10))
tree.plot_tree(best_tree,feature_names=df_clean.columns[0:-1], filled=True)

cm = confusion_matrix(y_test, y_pred, labels=best_tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=best_tree.classes_)
disp.plot()

plt.show()

