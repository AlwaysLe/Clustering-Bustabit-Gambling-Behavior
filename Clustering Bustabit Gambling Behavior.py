# Load
import pandas as pd
from scipy.cluster.vq import kmeans, vq
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn.decomposition import PCA
#Set the Max columns to display in Pandas
pd.set_option('display.max_columns', 15)

#Taks 1
# Read in the bustabit gambling data
url = 'https://raw.githubusercontent.com/erichare/projects-instructor-application-r/master/datasets/bustabit_sub.csv'
bustabit = pd.read_csv(url)
# Look at the first five rows of the data
bustabit.head(5)
# Find the highest multiplier (BustedAt value) achieved in a game
#bustabit['BustedAt'].max()
bustabit[bustabit['BustedAt']==bustabit['BustedAt'].max()]

#Task 2
# Create the new feature variables
bustabit['CashedOut'] = bustabit['CashedOut'].fillna(0.01+bustabit['BustedAt'])
bustabit.loc[bustabit['Profit']>0,'Losses'] = 0
bustabit['Losses'] = bustabit['Losses'].fillna(-bustabit['Bet'])
bustabit['Profit'] = bustabit['Profit'].fillna(0)
bustabit['GameWon'] = (bustabit['Profit']>0).astype('int')
bustabit['GameLost'] = (bustabit['Losses']<0).astype('int')
# Look at the first five rows of the features data
bustabit.head(5)

#Task 3
# Group by players to create per-player summary statistics
bustabit_clus = bustabit.groupby('Username').agg({'CashedOut': 'mean',
                                  'Bet': 'mean',
                                  'Profit': sum,
                                  'Losses': sum,
                                  'GameWon': sum,
                                  'GameLost': sum})
# View the first five rows of the data
bustabit_clus.head(5)

#Task 4
# Create the mean-sd standardization function
def mean_sd_standard(x):
    zscore = (x-x.mean())/x.std()
    return zscore
# Apply the function to each numeric variable in the clustering set
bustabit_standardized = bustabit_clus.apply(lambda x: mean_sd_standard(x), axis=0)
# Summarize our standardized data
bustabit_standardized.describe()


#Task 5
# Choose 20190101 as our random seed
np.random.seed(20190101)
# Cluster the players using kmeans with five clusters
cluster_solution, _1 = kmeans(bustabit_standardized, 5)
# Store the cluster assignments back into the clustering data frame object
bustabit_standardized['cluster'],_ =vq(bustabit_standardized, cluster_solution)
# Look at the distribution of cluster assignments
table = bustabit_standardized.groupby('cluster')['cluster'].count()

#Task 6
# Group by the cluster assignment and calculate averages
bustabit_clus['cluster'] = bustabit_standardized['cluster']
bustabit_clus_avg = bustabit_clus.groupby('cluster').mean()
# View the resulting table
bustabit_clus_avg

#Task 7
# Create the min-max scaling function
def min_max_standard(x):
    return ((x-min(x))/(max(x)-min(x)))
# Apply this function to each numeric variable in the bustabit_clus_avg object
bustabit_avg_minmax = bustabit_clus_avg.apply(lambda x: min_max_standard(x))
bustabit_avg_minmax = bustabit_avg_minmax.reset_index()
# Load the GGally package
#Down in the front
# Create a parallel coordinate plot of the values
#data = sns.load_dataset(bustabit_avg_minmax)
bustabit_avg_minmax = bustabit_avg_minmax[['cluster','Bet','Profit', 'Losses','CashedOut','GameWon','GameLost']]

pc = parallel_coordinates(bustabit_avg_minmax,
                          'cluster',
                          sort_labels = True,
                          color=('red', 'blue','purple','green','yellow'))

#Task 8
# Calculate the principal components of the standardized data
model = PCA()
samples = bustabit_standardized.drop(columns=['cluster'])
model.fit(samples)
my_pc = model.transform(samples)
my_pc = pd.DataFrame(my_pc, index=bustabit_standardized.drop(columns=['cluster']).index)
# Store the cluster assignments in the new data frame
my_pc['cluster'] = bustabit_standardized['cluster']
plt.bar(range(0,6),model.explained_variance_)

# Use ggplot() to plot PC2 vs PC1, and color by the cluster assignment
# View the resulting plot
fig, (p1, p2, p3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))


sns.scatterplot(my_pc[0],my_pc[1], hue = my_pc['cluster'], ax = p1,
                palette='tab10')
#p1.scatter(my_pc[0],my_pc[1], c = my_pc['cluster'], label = my_pc['cluster'])
p1.set_title('PC1 vs PC2')
sns.scatterplot(my_pc[1],my_pc[2], hue = my_pc['cluster'], ax = p2,
                palette='tab10')
#p2.scatter(my_pc[1],my_pc[2], c = my_pc['cluster'])
p2.set_title('PC2 vs PC3')
sns.scatterplot(my_pc[0],my_pc[2], hue = my_pc['cluster'], ax = p3,
                palette='tab10')
#p3.scatter(my_pc[0],my_pc[2], c = my_pc['cluster'])
p3.set_title('PC1 vs PC3')

plt.tight_layout()
plt.show()
#Task 9
# Assign cluster names to clusters 1 through 5 in order
cluster_names = {0:'Risky Commoners',1:'High Rollers',2:'Risk Takers',3:'Cautious Commoners',4:'Stratigic Addicts'}
# Append the cluster names to the cluster means table
bustabit_clus_avg_named = bustabit_clus_avg.rename(cluster_names)
# View the cluster means table with your appended cluster names
bustabit_clus_avg_named