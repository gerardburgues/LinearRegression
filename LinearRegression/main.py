"""
La nostra base de dades tracta sobre el rendiment d’alumnes de secundària en dos escoles portugueses.
 Els atributs inclueixen dades sobre les seves calificacions,
 característiques demogràfiques, socials i característiques relacionades amb l’escola.
  Totes aquestes dades han sigut obtingudes de informes escolars i qüestionaris.
"""

# Import Libraries

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



def DataInfo():
    """This function give us information about our table and our data
    shape, head, description, null values..."""
    print("We see all the variables that we have in our dataset: \n", data.head)

    print("What shape our CSV has ? \n",data.shape)

    print("Detailed information: \n", data.describe)

    print("Null values ? \n", data.isnull().sum())

def DropColumns(data):
    """We are droping those columns which we think are not worth having
    to generate a good prediction.
    """
    return data.drop(
        ['school', 'famsize', 'Pstatus', 'Fedu', 'Medu', 'Fjob', 'Mjob', 'reason', 'guardian', 'traveltime', 'famsup',
         'nursery', 'internet', 'goout', 'Dalc'], axis=1)
def plotsRelation(data):
    """This functions will show us the relation between each dataset component
    (Those variables we have deleted).
     """
    sns.pairplot(data)
    plt.show()
    print("unique? -->",data.nunique())
    print("new data head: \n", data.head)

def HeatMap(data):

    """Showing the heatmap from
    1. Can be all the data
    2. Can be just those specific rows (Not counting eliminated rows).
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 12), )

    ax = sns.heatmap(data=data.corr(), ax=ax, annot=True, cmap="coolwarm")
    ax.set_xlabel('Features', fontdict={"fontsize": 16})
    ax.set_ylabel('Features', fontdict={"fontsize": 16})
    for _, s in ax.spines.items():
        s.set_linewidth(5)
        s.set_color('cyan')
    ax.set_title('Correlation between different Features', loc="center",
                 fontdict={"fontsize": 16, "fontweight": "bold", "color": "white"}, )
    plt.savefig("plotcorrelation.png", bbox_inches="tight")
    plt.show()

def Mse(v1, v2):
    """ Apply MSE formula"""
    return ((v1 - v2)**2).mean()

def Regression(x, y):

    """ Apply the Regression methods with libary"""
    # Creem un objecte de regressió de sklearn

    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def TransformingStrings(data):

    """
    We have to transform data so we don't do the model with a character inside our dataset.
    """
    # No --> 0
    # Yes -->1
    data['schoolsup'] = data['schoolsup'].replace(['no'], 0)
    data['schoolsup'] = data['schoolsup'].replace(['yes'], 1)
    data['sex'] = data['sex'].replace(['M'], 0)
    data['sex'] = data['sex'].replace(['F'], 1)
    data['address'] = data['address'].replace(['U'], 0)
    data['address'] = data['address'].replace(['R'], 1)
    data['paid'] = data['paid'].replace(['no'], 0)
    data['paid'] = data['paid'].replace(['yes'], 1)
    data['activities'] = data['activities'].replace(['no'], 0)
    data['activities'] = data['activities'].replace(['yes'], 1)
    data['romantic'] = data['romantic'].replace(['no'], 0)
    data['romantic'] = data['romantic'].replace(['yes'], 1)
    data['higher'] = data['higher'].replace(['no'], 0)
    data['higher'] = data['higher'].replace(['yes'], 1)
    return data

if __name__ == "__main__":
    # Lets Read the data from our csv
    data = pd.read_csv("student-mat.csv")
    DataInfo()
    data = DropColumns(data)
    plotsRelation(data)
    HeatMap(data)
    data = TransformingStrings(data)
    print(data)
    #Values x and y
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("HEY   <",x)
    print(y)





else:
    print("File one executed when imported")

print("hi")
