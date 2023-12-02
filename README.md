# chronic-kidney-disease-prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing


data=pd.read_csv('/content/sample_data/kidney_disease.csv')
data

data.shape

data.info()

data.head()

data.tail()

data.describe()

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
categorical_cols = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']
encoder = OneHotEncoder(sparse=False, drop='first')  # 'drop' parameter removes one of the one-hot encoded columns to avoid multicollinearity
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
data = pd.concat([data, encoded_cols], axis=1)
data.drop(categorical_cols, axis=1, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

from sklearn.preprocessing import LabelEncoder

# Assuming 'df' is your DataFrame and 'target_column' is the name of your target variable column
target_column = 'classification'

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the target variable
data[target_column] = label_encoder.fit_transform(data[target_column])

data["dm_yes"].value_counts()

data["cad_no"].value_counts()

# visualization
plt.plot(data['dm_yes'])
plt.xlabel("dm")
plt.ylabel("Levels")
plt.title("dm levels Line Plot")
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
newdataset_len=data[data['classification']==1]['hemo'].value_counts()

ax1.hist(newdataset_len,color='red')
ax1.set_title('Having ckd')

newdataset_len=data[data['classification']==0]['hemo'].value_counts()
ax2.hist(newdataset_len,color='green')
ax2.set_title('NOT Having ckd')

fig.suptitle('hemo Levels')
plt.show()

data.duplicated()

newdataset=data.drop_duplicates()
newdataset

data.isnull().sum()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Assuming 'target_column' is the name of your target variable
X_train, X_test, Y_train, Y_test = train_test_split(data, data['classification'], test_size=0.2, random_state=42)

data

import pandas as pd

# Assuming 'df' is your DataFrame
feature_names = data.columns.tolist()

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

# Assuming 'classification' is a variable containing the target column name
classification = 'classification'  # Replace with your actual target column name

# Select features (X) and target variable (y)
feature_columns = ['id', 'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'pcc_present', 'pcc_nan', 'rbc_normal', 'rbc_nan', 'pc_normal', 'pc_nan', 'ba_present', 'ba_nan', 'htn_yes', 'dm_\tyes', 'dm_ yes', 'dm_no', 'dm_yes', 'cad_no', 'cad_yes', 'appet_poor', 'pe_yes', 'ane_yes']
X = data[feature_columns]
y = data[classification]

# Replace '\t?' with NaN
X.replace('\t?', np.nan, inplace=True)

# Convert columns to numeric (assuming that they are numeric features)
X = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
predictions = model.predict(test_X)

# Evaluate the model
accuracy = metrics.accuracy_score(predictions, test_Y)
print('The accuracy of the Logistic Regression model is:', accuracy)

# Display the classification report
report = classification_report(test_Y, predictions)
print("Classification Report:\n", report)


import matplotlib.pyplot as plt
import numpy as np

# Replace these values with your actual scores
precision = [0.79, 0.73]
recall = [0.89, 0.56]
f1_score = [0.83, 0.63]

labels = ['Class 0', 'Class 1']

# Plotting the bar chart
width = 0.2
x = np.arange(len(labels))

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Adding labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Logistic Regression Model Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Display the plot
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

# Assuming train_X, train_Y, test_X, test_Y are your training and testing data

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
train_X = imputer.fit_transform(train_X)
test_X = imputer.transform(test_X)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(train_X, train_Y)

# Make predictions on the test set
predictions = model.predict(test_X)



# Convert predictions to discrete classes (assuming a classification scenario)
predictions_classes = np.round(predictions).astype(int)

# Evaluate the model using accuracy (not typical for linear regression)
accuracy = accuracy_score(test_Y, predictions_classes)

print('Accuracy:', accuracy)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

# Assuming X, y are your features and target variable

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
train_X = imputer.fit_transform(train_X)
test_X = imputer.transform(test_X)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(train_X, train_Y)

# Make predictions on the test set
predictions = model.predict(test_X)

# Evaluate the model using mean absolute error
mae = mean_absolute_error(test_Y, predictions)
print('Mean Absolute Error:', mae)

# Evaluate the model using mean squared error
mse = mean_squared_error(test_Y, predictions)
print('Mean Squared Error:', mse)

# Evaluate the model using root mean squared error
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Evaluate the model using R-squared
r_squared = r2_score(test_Y, predictions)
print('R-squared:', r_squared)


