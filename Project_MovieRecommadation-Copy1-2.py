#!/usr/bin/env python
# coding: utf-8

# # Project: Movie Recommadation

# ## Step 1: Extraction of data 
# 
# Extract data from Web Scarping of IMDB one of the most famous website for movies

# #### Import library for web Scraping

# In[1]:


from bs4 import BeautifulSoup 
import requests
import csv


# In[2]:


# Define the number of pages you want to scrape
num_pages_to_scrape = 20


# #### Extract data and save it in a csv file named as imbd_movies.csv

# In[3]:


filename = "imdb_movies.csv"
headers = ["Name", "Year", "Runtime", "Genre", "Rating", "Votes", "Director", "Gross","MetaScore"]

with open(filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    base_url = "https://www.imdb.com/search/title?title_type=feature&year=2000-01-01,2023-12-31&sort=num_votes,desc&start=0&ref_=adv_nxt"
    page_no = 0;
    for page_number in range(1, num_pages_to_scrape + 1):
        # Build the URL for the current page
        
        url = f"https://www.imdb.com/search/title/?title_type=feature&year=2000-01-01,2023-12-31&sort=num_votes,desc&start={page_no*50+1}&ref_=adv_nxt"
        page_no = page_no +1 
        # Send an HTTP request to the page
        response = requests.get(url)


        if response.status_code == 200:
            
            # Parse the HTML content
            page_soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the containers (your code to do this)
            containers = page_soup.findAll("div",{"class":"lister-item mode-advanced"})


            for container in containers:

                # Extract Name
                name = container.find('img')['alt']

                # Extract year
                year_mov = container.find('span', {'class': 'lister-item-year'})
                year = year_mov.get_text(strip=True)

                # Extract Runtime
                runtime_mov = container.find('span', {'class': 'runtime'})
                if runtime_mov:
                    runtime = runtime_mov.get_text(strip=True)
                else:
                    runtime = ""


                # Extract Genre
                genre_mov = container.find('span', {'class': 'genre'})
                genre = genre_mov.get_text(strip=True)

                # Extract Rating
                rating_mov = container.find('div', {'class': 'ratings-imdb-rating'})
                rating = rating_mov.strong.get_text(strip=True)

                # Extract Votes
                votes_mov = container.find('span', {'name': 'nv'})
                votes = votes_mov['data-value']

                # Extract director
                director = container.find_all('p')[2].find('a').get_text()

                # Extract gross
                gross_mov = container.find('span', text="Gross:")
                if gross_mov:
                    gross = gross_mov.find_next('span', {'name': 'nv'})['data-value']
                else:
                    gross = ""

                
                #Extract MetaScore
                meta_mov = container.find('span',{'class':'metascore favorable'})
                meta = meta_mov.text if meta_mov else "N/A"

                # Print the data
                print(name, year, runtime, genre, rating, votes, director, gross,meta)

                # Write the data to the CSV file
                writer.writerow([name, year, runtime, genre, rating, votes, director, gross,meta])


                # Write the data to the CSV file

        else:
            print(f"Failed to retrieve data from page {page_number}")


# ## Step 2: Reading and Understanding the Data
# 
# Let's start with the following steps:
# 
# 1. Importing data using the pandas library
# 2. Understanding the structure of the data
# 3. Data Cleaning

# #### Import Library 

# In[156]:


import numpy as np
import pandas as pd


# #### Read data using pandas

# In[157]:


df = pd.read_csv("imdb_movies.csv" , sep=',')
df.sample(5)


# In[158]:


df.shape


# In[159]:


df.info()


# In[160]:


df.isnull().sum()


# In[161]:


df.describe()


# In[162]:


## copy data to make changes
df1 = df


# #### Fixing of Year column

# In[163]:


def Fixing(year):
    a=""
    for i in year:
        if(i.isnumeric()):
            a = a+i
        
    return a


# In[164]:


df1['Year'] = df1['Year'].apply(Fixing)


# In[165]:


df1.Year.value_counts()


# In[166]:


df1.Year.info()


# In[167]:


df1['Year']= df1['Year'].astype(int)


# In[168]:


df1.info()


# In[169]:


df1.head()


# #### Fixing of Runtime Column

# In[170]:


df1.Runtime = df1.Runtime.apply(Fixing)


# In[171]:


df1.Runtime = df1.Runtime.astype(float)


# In[172]:


df1.Runtime.dtype


# In[173]:


df1.sample(5)


# #### Fixing Gross

# In[174]:


df1.isnull().sum()


# In[175]:


df1 = df1.dropna(subset=['Gross'])


# In[176]:


df1.isnull().sum()


# In[177]:


df1.Gross.info()


# In[178]:


df1.Gross = df1.Gross.apply(Fixing)


# In[179]:


df1.Gross = df1.Gross.astype(float)


# In[180]:


df1.Gross.dtype


# In[181]:


df1.head()


# In[182]:


df_no_duplicates = df1.drop_duplicates(subset=['Name']).count()
print(df_no_duplicates)


# In[183]:


df1.drop_duplicates(inplace = True)


# In[184]:


df1.shape


# #### Fix MetaScore

# In[185]:


df1.isnull().sum()


# In[186]:


df1.MetaScore.value_counts()


# In[187]:


df1.MetaScore.median()


# In[188]:


df1.MetaScore = df1.MetaScore.fillna(df1.MetaScore.median())


# In[189]:


df1.isnull().sum()


# In[116]:


#### No such use of Director so drop them


# In[117]:


df1 = df1.drop('Director', axis=1)


# In[576]:


df1 = df1.drop('Name',axis = 1)


# In[190]:


df1.head()


# In[191]:


df1.describe()


# In[192]:


df1 = df1.dropna()


# ## Step 3: Visualising the Data

# ### Import some importatnt libraries help in data visualising

# In[193]:


import matplotlib.pyplot as plt 
import seaborn as sns


# In[194]:


plt.hist(df1['Rating'], bins=20, edgecolor='k')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Histogram of Rating')
plt.show()


# #### From above visualization i get to know that
# Most of movies having rating of 7 - 7.3

# In[195]:


plt.hist(df1['Votes'], bins=20, edgecolor='k')
plt.xlabel('Votes')
plt.ylabel('Frequency')
plt.title('Histogram of votes')
plt.show()


# In[196]:


# Create a pair plot for selected numerical columns
sns.pairplot(df1[['Rating', 'Votes', 'Gross', 'MetaScore','Runtime', 'Year']])
plt.show()


# #### Reviews from above scatterplot
# 
# 1. No negative relation.
# 2. 'Gross', 'MetaScore','Runtime', 'Year' have postive relation with target variable  Rating, Votes

# In[197]:


# Calculate the correlation matrix
corr_matrix = df1.corr()

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[198]:


plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(df1['Votes'], df1['Rating'], alpha=0.5)
plt.xlabel('Votes')
plt.ylabel('Rating')
plt.title('Scatter Plot: Votes vs. Rating')

plt.subplot(1, 3, 2)
plt.scatter(df1['MetaScore'], df1['Rating'], alpha=0.5)
plt.xlabel('MetaScore')
plt.ylabel('Rating')
plt.title('Scatter Plot: MetaScore vs. Rating')

plt.subplot(1, 3, 3)
plt.scatter(df1['Gross'], df1['Rating'], alpha=0.5)
plt.xlabel('Gross')
plt.ylabel('Rating')
plt.title('Scatter Plot: Gross vs. Rating')

plt.tight_layout()
plt.show()


# #### Review from Scatter plot
# 1. No negative relation
# 

# In[199]:


plt.figure(figsize = [8,8])
sns.boxplot(data=df1['Gross'])
plt.title("Gross Variable Distribution")
plt.ylabel("Gross")
plt.show()


# In[200]:


plt.figure(figsize = [8,8])
sns.boxplot(data=df1['Rating'])
plt.title("Rating Variable Distribution")
plt.ylabel("Rating")
plt.show()


# ## Step 4: Data Preparation

# In[201]:


genres = df1['Genre'].str.split(', ', expand=True)
genres = genres.stack().str.get_dummies().sum(level=0)


# In[202]:


# Concatenate the one-hot encoded genres with the original DataFrame
df1 = pd.concat([df1, genres], axis=1)


# In[203]:


df1.sample(5)


# In[204]:


# Select relevant columns for the model
selected_columns = ['Rating'] + list(genres.columns)
df2 = df1[selected_columns]


# In[205]:


df1['Rating'] = df1['Rating'].astype(float)


# In[206]:


df1.info()


# In[207]:


df2.info()


# ## Step 5: Splitting the Data into Training and Testing Sets
# As you know, the first basic step for regression is performing a train-test split.

# In[210]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[224]:


# Split the data into training and testing sets
X = df2.drop('Rating', axis=1)  # Independent variables (genres)
y = df2['Rating']  # Dependent variable (rating)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Step 6: Build a modal

# In[225]:


from sklearn.preprocessing import MinMaxScaler


# In[226]:


# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[227]:


# scaler = MinMaxScaler()


# In[228]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[229]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[230]:


plt.figure(figsize = (12, 6))
sns.heatmap(X_train.corr(),annot=True)
plt.show()


# In[231]:


# Choose a target year (e.g., 2022)
target_year = 2022

# Filter the movies of the target year
target_movies = df1[df1['Year'] == target_year]


# In[216]:


# Predict ratings for the target movies using numeric columns
predicted_ratings = model.predict(target_movies.drop(['Rating', 'Name'], axis=1))


# In[217]:



# Add the predicted ratings to the DataFrame
target_movies['Predicted_Rating'] = predicted_ratings

# Sort the target movies by predicted rating in descending order
target_movies = target_movies.sort_values(by='Predicted_Rating', ascending=False)

# Get the top recommended movie and its predicted rating
top_recommendation = target_movies.head(1)
if not top_recommendation.empty:
top_movie_name = top_recommendation['Name'].values[0]
top_movie_rating = top_recommendation['Predicted_Rating'].values[0]
print(f"Top Recommended Movie in the '{target_year}' year: '{top_movie_name}' with a predicted rating of {top_movie_rating:.2f}")
else:
print(f"No movie recommendations found in the '{target_year}' year.")



# In[ ]:




