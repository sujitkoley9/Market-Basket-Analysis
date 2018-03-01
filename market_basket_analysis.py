
##http://pbpython.com/market-basket-analysis.html
#https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
#http://www.salemmarafi.com/code/market-basket-analysis-with-r/

#----------------- Market basket Analysis 1: -------------------------------------#

#----Import package------------------------------------------------------------#
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#----Read the input file-------------------------------------------------------#
RequiredDf = pd.read_excel('Online Retail.xlsx')
RequiredDf.head

#-------Data preparation ------------------------------------------------------#
RequiredDf.isnull().sum()

RequiredDf.dropna(axis=0,subset=['InvoiceNo','Description'], inplace=True) 


    #----------Prepare basket for only france-------------------------------#
basket = (RequiredDf[RequiredDf['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))




def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

#-----------------------Applying algorithm -----------------------------------#
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

#--------------------Fetching Rules-------------------------------------------#
rules[ (rules['lift'] >= 6) &
       (rules['confidence'] >= 0.8) ]










#----Import package------------------------------------------------------------#
import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#----data preparation----------------------------------------------------------#

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]


oht = OnehotTransactions()
oht_ary = oht.fit(dataset).transform(dataset)
RequiredDf = pd.DataFrame(oht_ary, columns=oht.columns_)



#-----------------------Applying algorithm -----------------------------------#
frequent_itemsets = apriori(RequiredDf, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()

#--------------------Fetching Rules-------------------------------------------#
rules=rules[ (rules['lift'] >= 5) &
       (rules['confidence'] >= 0.9) ]

