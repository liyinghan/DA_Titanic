import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import randn
import sys
import json
import matplotlib as mlt
import matplotlib.pyplot as plt
from io import StringIO
import seaborn as sns
import scipy as stats

# read file and import the data
titanic_df = pd.read_csv('train.csv')
#titanic_df = sns.load_dataset("titanic")

# take a brief look at the data

'''
titanic_df.info()
'''

# Using questions to drive your analysis
# who are the passengers on titanic?understand who are passengers? their characteristics.
# use the summary or descriptive analysis to get a feeling of your data
# gender, age, mean, max, geo, race, etc..all those stuff


#sns.catplot(x="Sex", data=titanic_df)

#sns.catplot(x="Sex", data=titanic_df, hue="Pclass")
#sns.catplot(x='Pclass', data=titanic_df, hue='Sex')

#sns.factorplot('Pclass', data=titanic_df, hue='Sex')

#sns.factorplot("Pclass", data=titanic_df, hue="Sex")

def male_female_child(passenger):
    age, sex = passenger

    if age < 16:
        return 'child'
    else:
        return sex

titanic_df['person']= titanic_df[['Age','Sex']].apply(male_female_child, axis=1)

#sns.factorplot('Pclass', data=titanic_df, hue='person')
#sns.catplot(x='Pclass', data=titanic_df, hue='Sex')

'''
titanic_df['Age'].hist(bins=70)
print(titanic_df['Age'].mean())
person_counts = titanic_df['person'].value_counts()
print(person_counts)
'''


# print area to check if the output is correct
'''
print(type(titanic_df)) # output is dataframe

'''
# print(titanic_df.head(20))


####################-------part 2-------------##########

'''
fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()


fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_df['Age'].max()
fig.set(xlim=(0, oldest))
fig.add_legend()
'''

'''
deck = titanic_df['Cabin'].dropna()
print(deck.head(10))

levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']

sns.catplot(x='Cabin', data=cabin_df, kind='count', palette='Blues_d')

cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.catplot(x='Cabin', data=cabin_df, kind='count', palette='summer',
            order=['A', 'B', 'C', 'D', 'E', 'F', 'G'])


# where do people come from?
sns.catplot(x='Embarked', data=titanic_df, kind='count', hue='Pclass',
            order=['C', 'Q', 'S'])
'''

####################-------part 3-------------##########

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
#print(titanic_df['Alone'])

titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


#plot the chart
sns.catplot(x='Alone', data=titanic_df, kind='count', palette='Blues')

titanic_df['survivor'] = titanic_df.Survived.map({0: '0', 1: 'yes'})

sns.catplot(x='survivor', data=titanic_df, kind='count', palette='Set1')

sns.catplot(x='Pclass', y='Survived', kind='point', hue='person', data=titanic_df)

#sns.lmplot('Age', 'Survived', hue='Pclass',  data=titanic_df, palette='winter')

generations = [10, 20, 40, 60, 80]

sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_df, palette='winter',
           x_bins=generations)

sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df, palette='winter',
           x_bins=generations)


# show plot
plt.show()
