#########################################
#
# The following file contains all the
# higher order functions used throughout
# the notebooks, for visual smoothing
# of the notebook order.
#
#########################################


#### Modules ####

import numpy as np
import matplotlib.pyplot as plt

#### PLOTTING FUNCTIONS #####


def pair_keyword_type(keyword, dataframe): 
    dataarray = []
    for i in range(0, dataframe.__len__()): 
        instances = dataframe.content[i].count(str(keyword))
        dataarray.append((instances, dataframe.type[i]))
    return dataarray


def scatterplot_keyword_type(keyword, dataframe):
    pair_keyword_list = pair_keyword_type(keyword, dataframe)
    tuple_array = pair_keyword_type(keyword, dataframe)
    typearray = []
    countarray = []
    for i in range(0,pair_keyword_list.__len__()): 
        typearray.append(str(tuple_array[i][1]))
        countarray.append(tuple_array[i][0])
    fig = plt.figure(figsize=(4,4))
    ax =fig.add_axes([0,0,2,1])
    ax.scatter(typearray, countarray)
    plt.xticks(rotation = 'vertical')
    plt.show()


def barplot_keyword_type(keyword, dataframe): 
    tuple_array = pair_keyword_type(keyword, dataframe)
    rawdict = {}
    for i in range(0,len(tuple_array)):
        if tuple_array[i][1] not in rawdict:
            rawdict[tuple_array[i][1]] = [tuple_array[i][0]]
        else:
            rawdict[tuple_array[i][1]].append(tuple_array[i][0])

    meandict={}
    for i in rawdict: 
        meandict[i] = np.mean(rawdict[i])
    mediandict={}
    for i in rawdict: 
        mediandict[i] = np.median(rawdict[i])
    meanvals =[]
    medianvals=[]
    for i in mediandict:
        medianvals.append(mediandict[i])
    for i in meandict:
        meanvals.append(meandict[i])
    types = list(mediandict.keys())
    types = [str(i) for i in types]
    print(medianvals)
    print(meanvals)
    print(types)
    df = pd.DataFrame({
    'types': types,
    'meanvals': meanvals,
    'medianvals': medianvals
    })
    df.plot(x="types", y=["meanvals", "medianvals"], kind="bar", figsize=(10,5))


def boxplot_keyword_type(keyword, dataframe):
    tuple_array=pair_keyword_type(keyword, dataframe) 
    boxdata=[]
    dict = {}
    for i in range(0,len(tuple_array)):
        if tuple_array[i][1] not in dict:
            dict[tuple_array[i][1]] = [tuple_array[i][0]]
        else:
            dict[tuple_array[i][1]].append(tuple_array[i][0])
    for i in dict:
        boxdata.append(dict[i])
    fig = plt.figure()
    ax =fig.add_axes([0,0,2,1])
    ax.boxplot(boxdata, labels = list(dict.keys()))
    plt.xticks(rotation = 'vertical')
    plt.show()