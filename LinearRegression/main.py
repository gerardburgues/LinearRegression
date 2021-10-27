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
from sklearn.decomposition import PCA





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
        ['school', 'famsize', 'Pstatus', 'Fedu', 'Medu', 'Fjob', 'Mjob',
         'reason', 'guardian', 'traveltime', 'famsup',
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

    #Showing the specific correlation
    pairplot = sns.pairplot(data[["G1", "G2", "G3"]], palette="viridis")

    plt.savefig("pairplot.png", bbox_inches="tight")
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
def NormalizeData(data, value):

    """We normalize data to have the values between 0 and 1 """
    min_max = MinMaxScaler()
    dataNormalize = min_max.fit_transform(data)
    if value == 1:

        df = pd.DataFrame(dataNormalize, columns=['sex', 'age', 'address', 'studytime', 'failures',
                                              'schoolsup', 'paid', 'activities', 'higher',
                                              'romantic', 'famrel', ' freetime ', 'Walc', 'health', 'absences', 'G1',
                                              ' G2', 'G3'])
    else:

        df = pd.DataFrame(dataNormalize, columns=[
            'G1', 'G2', 'G3'])

    return df
def PlotNormalize(x_df):

    """Showing how many values in the variables G1 and G2 appears.
    Taking into account we have normalized.
    """
    x = x_df.to_numpy()
    #G2
    plt.figure()
    plt.title("Histograma G2")
    plt.xlabel("G2")
    plt.ylabel("Count")
    hist = plt.hist(x[:, 16], bins=11, range=[np.min(x[:, 0]), np.max(x[:, 0])], histtype="bar", rwidth=0.8)
    plt.show()
    ## G1
    plt.figure()
    plt.title("Histograma G1")
    plt.xlabel("G1")
    plt.ylabel("Count")
    hist = plt.hist(x[:, 15], bins=11, range=[np.min(x[:, 0]), np.max(x[:, 0])], histtype="bar", rwidth=0.8)

def RegressionAllVariables(x, y, value):
    """
    Applying the regression to all those we set in x and y.
    Returnning the MSE and R2_score
    """
    regr = Regression(x, y)
    predicted = regr.predict(x)

    x_df1 = x.to_numpy()
    plt.figure()
    # predicció de xx
    # s'ha de mutiplicar per xx per que es el range que li diem que ha de tirar
    # sino anira a tots.
    if value == 1:
        xx = np.arange(0, 1, 0.01)
        ax = plt.scatter(y, x_df1[:, 16])
        plt.plot(regr.coef_[16] * xx + regr.intercept_, xx, 'r')
    else:

        xx = np.arange(0, 1, 0.01)
        ax = plt.scatter(y, x['G2'])
        plt.plot(regr.coef_[1] * xx + regr.intercept_, xx, 'r')

    MSE = Mse(y, predicted)
    r2 = r2_score(y, predicted)

    plt.title("Regressió amb 'G2' i la predicció")
    plt.savefig("toteslesdades.png", bbox_inches="tight")
    print("Mean squeared error: ", MSE)
    print("R2 score: ", r2)

def split_data(x, y, train_ratio=0.85):
    """
    Spliting data 85% training 15% test

    """
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:]
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val
def RegressionMultiple(x_train, x_val,y_train,y_val):
    """

    :param x_train: Training values from x
    :param x_val: Test values from x
    :param y_train: Train values from y
    :param y_val:  Test values from y
    :return:  Returns the MSE and error of each atribute
    """
    for i in range(x_train.shape[1]):
        x_t = x_train[:, i]  # seleccionem atribut i en conjunt de train
        x_v = x_val[:, i]  # seleccionem atribut i en conjunt de val.
        x_t = np.reshape(x_t, (x_t.shape[0], 1))
        x_v = np.reshape(x_v, (x_v.shape[0], 1))

        regr = Regression(x_t, y_train)
        error = Mse(y_val, regr.predict(x_v))  # calculem error
        r2 = r2_score(y_val, regr.predict(x_v))

        print("Error en atribut %d: %f" % (i, error))
        print("R2 score en atribut %d: %f" % (i, r2))


if __name__ == "__main__":
    # Lets Read the data from our csv
    data = pd.read_csv("student-mat.csv")
    #DataInfo()
    data = DropColumns(data)
    plotsRelation(data)
    HeatMap(data)
    data = TransformingStrings(data)
    #All data 16 variables
    df = NormalizeData(data, 1)
    ##dividing data into atributes_variables and goal_variable
    x_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]
    PlotNormalize(x_df)
    RegressionAllVariables(x_df, y_df, 1)
    #2 variables 1 goal
    #Just taking those 3 variables and Normalizing
    data_total = data.iloc[:, [15, 16, 17]]
    df = NormalizeData(data_total, 0)
    x1_ = df.iloc[:, :-1]
    y1_ = df.iloc[:, -1]
    RegressionAllVariables(x1_, y1_, 0)
    #Split data  training test
    x_train, y_train, x_val, y_val = split_data(x1_.to_numpy(), y1_.to_numpy())
    #Applying regression with training and test
    RegressionMultiple(x_train, x_val, y_train, y_val)



else:
    print("File one executed when imported")


