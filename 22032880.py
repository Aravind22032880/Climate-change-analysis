#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from scipy.optimize import curve_fit
import wbgapi as wb
import pandas as pd
from sklearn.cluster import KMeans


def world(indicators, count):
    """ retrieving data from world bank api and transposing data"""
    world = wb.data.DataFrame(indicators, count, mrv=30)
    world_pd = pd.DataFrame(world.sum())
    world_t = world.transpose()
    return world, world_pd, world_t


def preprocess(df):
    year = []
    for i in df.index:
        x = int(i.strip("YR"))
        year.append(x)
    df["year"] = year
    df_index = df.set_index("year")

    return df_index



countries = ["CHN", "IND", "USA", "RUS", "JPN"]



indicators = ["EN.ATM.METH.KT.CE", "EN.ATM.CO2E.KT"]



meth, meth_pd, meth_t = world(indicators[0], countries)



meth_pdp = preprocess(meth_pd)



co2, co2_pd, co2_t = world(indicators[1], countries)



co2_pdp = preprocess(co2_pd)



def map_corr(df, size=6):
    """Function creates heatmap of correlation matrix for each pair of
    columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)

    The function does not have a plt.show() at the end so that the user
    can savethe figure.
    """

    corr = df.corr()
    plt.figure(figsize=(size, size))
    # fig, ax = plt.subplots()
    plt.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    plt.show()



"""correlation of methane emission over selected countries"""
map_corr(meth_t)


"""correlation of co2 emissions over selected countries"""
map_corr(co2_t)

# Normalising and back scaling of cluster centers:

def scaler(df):
    """ Expects a dataframe and normalises all
        columnsto the 0-1 range. It also returns
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


co2_x, co2_y, co2_z = scaler(co2_t)

meth_x, meth_y, meth_z = scaler(meth_t)

def backscale(df, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    arr = df.to_numpy()
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        #print(arr.type)
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr


backscaled_co2 = backscale(co2_x,co2_y,co2_z)

backscaled_meth = backscale(meth_x, meth_y, meth_z)

def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.

    This routine can be used in assignment programs.
    """

    import itertools as iter

    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower

    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))

    pmix = list(iter.product(*uplow))

    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper


def exp_growth(t, scale, growth):
    #referred to unit 9
    """Computes exponential function with scale and growth as free parameters"""
    f = scale * np.exp(growth * (t-2005))
    return f


def curve(year, data, z, x, y):
    """function that plot graph between confidence ranges and fit """
    popt, pcov = curve_fit(exp_growth, year, data)
    z["pop_exp"] = exp_growth(year, *popt)
    sigma = np.sqrt(np.diag(pcov))
    low, up = err_ranges(year, exp_growth, popt, sigma)
    plt.figure()
    plt.title("exp_growth function")
    plt.plot(year, data, "o", markersize=5, label=x)
    plt.plot(year, z["pop_exp"], label="fit")
    plt.fill_between(year, low, up, alpha=1)
    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

methane = ["year", "Methane emissions"]

co2 = ["year", "co2"]



curve_fit_methane = curve(meth_pdp.index, meth_pdp[0],
                          meth_pdp, methane[0], methane[1])




curve_fit_co2 = curve(co2_pdp.index, co2_pdp[0], co2_pdp, co2[0], co2[1])




def get_diff_entries(df1, df2, column):
    """ Compares the values of column in df1 and the column with the same
    name in df2. A list of mismatching entries is returned. The list will be
    empty if all entries match. """

    import pandas as pd

    # merge keeping only rows in common
    diff_list = pd.merge(df1, df2, on=column, how="inner")

    return diff_list




whole_data = get_diff_entries(meth_pdp[0], co2_pdp[0], meth_pdp.index)

kmeanModel = KMeans(n_clusters=3)
kmeans_fit = kmeanModel.fit_predict(whole_data[['0_x', '0_y']])
cluster_centers = kmeanModel.cluster_centers_

u_labels = np.unique(kmeans_fit)

cluster = whole_data[['0_x', '0_y']].copy()

cluster['Clusters'] = kmeans_fit

fig = plt.figure(figsize=(5, 5))

plt.scatter(cluster['0_x'], cluster['0_y'],
            c=cluster['Clusters'], cmap='viridis')


plt.title("Scatter plot")
plt.xlabel('Methane emissions')
plt.ylabel('Co2 emissions')
plt.show()
