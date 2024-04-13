#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# Customer personality analysis can be used to segment customers into different groups based on their demographics, psychographics, and behavior. This information can then be used to create targeted marketing campaigns and improve customer service.
# 
# ID: Customer's unique identifier
# **Year_Birth:** Customer's birth year
# **Education:** Customer's education level
# **Marital_Status:** Customer's marital status
# **Income:** Customer's yearly household income
# **Kidhome:** Number of children in customer's household
# **Teenhome:** Number of teenagers in customer's household
# **Dt_Customer:** Date of customer's enrollment with the company
# **Recency:** Number of days since customer's last purchase
# **Complain:** 1 if the customer complained in the last 2 years, 0 otherwise
# Products
# 
# **MntWines:** Amount spent on wine in last 2 years
# **MntFruits:** Amount spent on fruits in last 2 years
# **MntMeatProducts:** Amount spent on meat in last 2 years
# **MntFishProducts:** Amount spent on fish in last 2 years
# **MntSweetProducts:** Amount spent on sweets in last 2 years
# **MntGoldProds:** Amount spent on gold in last 2 years
# Promotion
# 
# **NumDealsPurchases:** Number of purchases made with a discount
# **AcceptedCmp1:** 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# **AcceptedCmp2:** 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# **AcceptedCmp3:** 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# **AcceptedCmp4:** 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# **AcceptedCmp5:** 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# **Response:** 1 if customer accepted the offer in the last campaign, 0 otherwise
# Place
# 
# **NumWebPurchases:** Number of purchases made through the company’s website
# **NumCatalogPurchases:** Number of purchases made using a catalogue
# **NumStorePurchases:** Number of purchases made directly in stores
# **NumWebVisitsMonth:** Number of visits to company’s website in the last month

# In[1]:


#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# In[2]:


#load the dataset
marketing_data = pd.read_csv("/kaggle/input/customer-personality-analysis/marketing_campaign.csv",header =0,sep='\t')
marketing_data.head()


# In[3]:


#dimension
marketing_data.shape


# In[4]:


#checking null values

msno.matrix(marketing_data)
plt.show()

marketing_data.isna().sum()


# In[5]:


#fill up the null value in income column
print(marketing_data['Income'].dtypes)
marketing_data['Income'] = marketing_data['Income'].fillna(marketing_data['Income'].median())
marketing_data.isna().sum()


# In[6]:


marketing_data.info()


# # **EDA**

# In[7]:


marketing_data.columns


# In[8]:


#categorical Distribution
#sns.countplot(data=marketing_data,x='Education')
Number_of_diff_eductaionbackground_consumers = marketing_data['Education'].value_counts()
Number_of_diff_eductaionbackground_consumers.plot(kind='pie',autopct="%.2f%%",explode=[0.01,0.02,0.03,0.05,0.05],shadow=True)
plt.title("percentage of consumers from different education level")
plt.show()


# In[9]:


#bar chart
sns.countplot(data=marketing_data,x='Education',hue="Marital_Status")
plt.title("Marital Status of different education lvl cosumers")
plt.ylabel("Frequency")
plt.show()


# **Maximum consumers are married here. than live together**.  

# In[10]:


#income statistics
print(marketing_data['Income'].describe())

#income Distribution
plt.figure(figsize=(16, 6)) 
plt.subplot(1,2,1)
sns.distplot(a=marketing_data['Income'],bins=50)
plt.title("Income Distribution of consumers")

plt.subplot(1,2,2)
sns.barplot(data=marketing_data,x="Education",y="Income",palette="viridis")
plt.title("Income of the consumers from diff Educational Background")
plt.tight_layout()
plt.show()


# **PhD holders have high rank of Income**

# In[11]:


#income statistics
print(marketing_data['Income'].describe())

plt.figure(figsize=(16,6))

#spent on wine in last 2 years
plt.subplot(1,2,1)
sns.scatterplot(data=marketing_data,x="Income",y="MntWines",hue="Education",style="Complain")
plt.title("spent on Wine in last 2 year either complain or not")
plt.xlim(0,120000)

#spent on fruits in last 2 years
plt.subplot(1,2,2)
sns.scatterplot(data=marketing_data,x="Income",y="MntFruits",hue="Education",style="Complain")
plt.title("spent on Fruits in last 2 year either complain or not")
plt.xlim(0,120000)


# In[12]:


plt.figure(figsize=(16,6))
#spent on meat in last 2 years
plt.subplot(1,2,1)
sns.scatterplot(data=marketing_data,x="Income",y="MntMeatProducts",hue="Education",style="Complain",palette="magma")
plt.title("spent on meat in last 2 year either complain or not")
plt.xlim(0,120000)

#spent on fish products 
plt.subplot(1,2,2)
sns.scatterplot(data=marketing_data,x="Income",y="MntFishProducts",hue="Education",style="Complain",palette="BuPu")
plt.title("spent on Fish in last 2 year either complain or not")
plt.xlim(0,120000)

plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(16,6))
#spent on meat in last 2 years
plt.subplot(1,2,1)
sns.scatterplot(data=marketing_data,x="Income",y="MntSweetProducts",hue="Education",style="Complain",palette="ocean")
plt.title("spent on sweet in last 2 year either complain or not")
plt.xlim(0,120000)

#spent on fish products 
plt.subplot(1,2,2)
sns.scatterplot(data=marketing_data,x="Income",y="MntGoldProds",hue="Education",style="Complain",palette="viridis")
plt.title("spent on gold in last 2 year either complain or not")
plt.xlim(0,120000)

plt.tight_layout()
plt.show()


# In[14]:


#. Each box in the plot represents the interquartile range (IQR) of the wine and fruits spending for a specific education level,
#with the median indicated by the horizontal line inside the box.

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.boxplot(data=marketing_data,x="Education",y="MntWines")
plt.title("Amount spent on wines across different level of education")

#Fruits
plt.subplot(1,2,2)
sns.boxplot(data=marketing_data,x="Education",y="MntFruits")
plt.title("Amount spent on Fruits across different level of education")
plt.tight_layout()
plt.show()


# # # **Each box in the plot represents the interquartile range (IQR) of the wine and fruits spending for a specific education level,with the median indicated by the horizontal line inside the box. Outliers are also detected.**

# In[15]:


#meat
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.boxplot(data=marketing_data,x="Education",y="MntMeatProducts")
plt.title("Amount spent on meat across different level of education")

#Fish
plt.subplot(1,2,2)
sns.boxplot(data=marketing_data,x="Education",y="MntFishProducts")
plt.title("Amount spent on Fish across different level of education")
plt.tight_layout()
plt.show()


# In[16]:


#Sweetroducts
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.boxplot(data=marketing_data,x="Education",y="MntSweetProducts")
plt.title("Amount spent on sweet across different level of education")

#Gold
plt.subplot(1,2,2)
sns.boxplot(data=marketing_data,x="Education",y="MntGoldProds")
plt.title("Amount spent on gold across different level of education")
plt.tight_layout()
plt.show()


# In[17]:


#list of categorical features
category = marketing_data.select_dtypes(include="object").columns
print(category)


# # **Purchase from website,catalog,store**

# In[18]:


#making new dataframes 
NumDealPurchase = marketing_data.groupby("Education")["NumDealsPurchases"].agg("sum").sort_values(ascending=False).reset_index()
NumWebPurchase = marketing_data.groupby("Education")['NumWebPurchases'].agg("sum").sort_values(ascending=False).reset_index()
NumcatalogPurchase = marketing_data.groupby("Education")["NumCatalogPurchases"].agg("sum").sort_values(ascending=False).reset_index()
NumStorePurchase = marketing_data.groupby("Education")['NumStorePurchases'].agg("sum").sort_values(ascending=False).reset_index()


# In[19]:


from IPython.display import display

# Concatenate DataFrames horizontally
df_combined = pd.concat([NumDealPurchase, NumWebPurchase, NumcatalogPurchase, NumStorePurchase], axis=1)

# Display the combined DataFrame
display(df_combined)


# In[20]:


plt.figure(figsize=(16,8))

plt.subplot(2,2,1)
sns.barplot(data = NumDealPurchase,x="Education",y="NumDealsPurchases",palette="viridis")
plt.title("Total number of purchases made with a discount across different level of education")

plt.subplot(2,2,2)
sns.barplot(data = NumWebPurchase,x="Education",y="NumWebPurchases",palette="magma")
plt.title("Total number of purchases from website across different level of education")

plt.subplot(2,2,3)
sns.barplot(data = NumcatalogPurchase,x="Education",y="NumCatalogPurchases")
plt.title("Total number of purchases from catalog  across different level of education")

plt.subplot(2,2,4)
sns.barplot(data = NumStorePurchase,x="Education",y="NumStorePurchases",palette="ocean_r")
plt.title("Total number of purchases from store directly across different level of education")


plt.tight_layout()
plt.show()


# In[21]:


sns.barplot(data=marketing_data,x='Education',y='NumWebVisitsMonth')
plt.title("Number of visits to the company's website last month")
plt.show()


# # **Well ! It's shocking but the bar chart say's that consumer's from basic education level visit most of the time than others although in previous cases the response from Basic level consumers were minimal. **

# In[22]:


ax = sns.countplot(data=marketing_data,x='Complain',hue="Education")

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha = 'center', va = 'center', 
                 xytext = (0, 10), 
                 textcoords = 'offset points')
    
ax.set_xticklabels(['No Complaint', 'Complaint'])
plt.title("Consumers complaint in last 2 years")
plt.show()


# ## **It seems clearly that most of the consumers are satisfied with the products and services. No complain from consumer side. **

# # **Correlation**

# In[23]:


#correlation between the features itselfs which explain the linear relationship between them

df = marketing_data.drop(["ID","Year_Birth","Education","Marital_Status","Dt_Customer","Z_Revenue","Z_CostContact","Response"],axis=1)
plt.figure(figsize=(16,6))
sns.heatmap(data=df.corr(),annot=True)
plt.show()


# In[24]:


#positive and negative correlations of the Income with other feature

df_corr = df.corr()
plt.figure(figsize = (10,6))
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
df_corr['Income'].sort_values(ascending = False).plot(kind = 'bar',color="red")
plt.title("Positive and negative correlations of the features with Income")
plt.xlabel("Features")
plt.ylabel("correlation coafficient")
plt.show()


# # **Market compagion offer**

# In[25]:


market_compagion = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']
plt.figure(figsize=(16, 6))

for i, column in enumerate(market_compagion, 1):
    plt.subplot(2, 3, i)
    ax = sns.countplot(data=marketing_data, x=column,palette="viridis")

    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    ax.set_xticklabels(["Not accepted","accepted"])

plt.suptitle("Acceptance of Offers in Marketing Campaigns")
plt.tight_layout()
plt.show()


# ### **Most of the consumers rejected the offer.**

# # **LabelEncoding**

# In[26]:


#convert categorical column into numerical
lbl = LabelEncoder()
std = StandardScaler()
for i in category:
    marketing_data[i] = lbl.fit_transform(marketing_data[i])


# In[27]:


#overview
marketing_data.head()


# In[28]:


#split the feature and target

X = marketing_data.drop(['Response','Z_Revenue','Z_CostContact','ID'],axis=1)
y = marketing_data['Response']

#standarization(z-scaling)
std_x = std.fit_transform(X)
X_train, X_test,y_train,y_test = train_test_split(std_x,y,test_size=0.2)


# # **SMOTE + RandomForest**

# In[29]:


from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Fit the classifier on the resampled training data
rf_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy:{accuracy*100:.3f} %")


# In[30]:


#get the important features

feature_importances = rf_classifier.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


# In[31]:


plt.figure(figsize=(8, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='green')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.show()


# # **PCA+SMOTE+RandomForest**

# In[32]:


from sklearn.decomposition import PCA

X_tr, X_tst, y_tr, y_tst = train_test_split(X, y, test_size=0.2, random_state=32)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_tr)
X_test_scaled = scaler.transform(X_tst)

# Apply PCA to reduce the dimensionality
pca = PCA(n_components=0.97)  # Keep 95% of the variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Apply SMOTE to balance the training data
smote = SMOTE()
X_resampled_pca, y_resampled_pca = smote.fit_resample(X_train_pca, y_train)

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the classifier on the resampled training data
rf.fit(X_resampled_pca, y_resampled_pca)

# Make predictions on the test set
y_pred = rf.predict(X_test_pca)

# Calculate the accuracy score
accuracy_pca = accuracy_score(y_test, y_pred)

print(f"Accuracy:{accuracy_pca*100:.4f} %")


# In[33]:


#trying XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_resampled,y_resampled)
print(f"Accuracy: {xgb.score(X_test,y_test)*100:.4f} %")


# In[38]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Apply PCA to reduce the dimensionality to 2 or 3
#pca = PCA(n_components=2)  # Choose the number of components
#X_pca = pca.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# In[46]:


from sklearn.cluster import KMeans
import numpy as np

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit K-means clustering on the dataset
kmeans.fit(X)

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Get cluster centers
cluster_centers = kmeans.cluster_centers_

# Print cluster centers
print("Cluster Centers:")
print(cluster_centers)

# Assign each data point to its corresponding cluster
clustered_data = np.column_stack((X, cluster_labels))

# Print the first few rows of the clustered data
print("Clustered Data:")
print(clustered_data[:5])


# In[47]:


import matplotlib.pyplot as plt

# Plotting the clusters
plt.figure(figsize=(8, 6))

# Iterate through each cluster and plot its data points
for cluster_id in range(n_clusters):
    # Select data points belonging to the current cluster
    cluster_data = X[cluster_labels == cluster_id]
    
    # Plot the data points for the current cluster
    plt.scatter(cluster_data.iloc[:, 3], cluster_data.iloc[:, 8], label=f'Cluster {cluster_id + 1}')

# Plotting the cluster centers
plt.scatter(cluster_centers[:, 3], cluster_centers[:, 8], marker='x', color='red', label='Cluster Centers')

# Add labels and title
plt.xlabel('Income')
plt.ylabel('MntWines')
plt.title('K-means Clustering')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

