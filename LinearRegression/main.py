"""This data approach student achievement in secondary education of two Portuguese schools.
The data attributes include student grades, demographic, social and school related features)
and it was collected by using school reports and questionnaires.
Two datasets are provided regarding the performance in two distinct subjects:
Mathematics (mat) and Portuguese language (por).
In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks.
Important note: the target attribute G3 has a strong correlation with attributes G2 and G1.
This occurs because G3 is the final year grade (issued at the 3rd period),
 while G1 and G2 correspond to the 1st and 2nd period grades.
 It is more difficult to predict G3 without G2 and G1, but such prediction
 is much more useful (see paper source for more details)."""



from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Reading data in CSV files
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# loading example into a dataset --> data
dataset = load_dataset('student-mat.csv')
data = dataset.values
print(dataset)
x = data[:, :2]
y = data[:, 2]

print("BBDD size:", dataset.shape)
print("X dimension:", x.shape)
print("Y dimension: ", y.shape)

#We show that there is no data with nan or nulls elements
df = pd.DataFrame(data)
print(df.isnull().values.any())
print(dataset.isnull().sum())

##Show numeric data


import seaborn as sns

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()

ax = sns.scatterplot(x='G3', y='age', data= dataset)
ax1 = sns.scatterplot(x='G1', y='G3', data= dataset)
plt.show()