#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import numpy as np

def policy_mine():
    # Send a GET request to the webpage
    url = "https://www.whitehouse.gov/climate/"
    response = requests.get(url)
    
    
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
    else: 
        print("Error making request, try after some time")
    
    target_element = soup.find_all("div", class_="wysiwyg wysiwyg-text acctext--con")
    target_elements=target_element[6:]
    headline_elements = soup.find_all("h3", class_="accordion__headline")
    headings= [i.text.strip().replace("\xa0", " ") for i in headline_elements]
    
    
    data=[]
    
    # Iterate through the HTML data for each sector
    for html_data, sector in zip(target_elements, headings):
        ul_element = html_data.find("ul")
        
        # Check if the ul element exists
        if ul_element:
            # Find all list items (li) under the ul element
            list_items = ul_element.find_all("li")
            
            # Extract text from each li item
            #li_texts = [item.get_text(strip=True) for item in list_items]
            li_texts = [item.text.strip() for item in list_items]
            
            # Append the data to the list
            data.extend([(sector, li_text) for li_text in li_texts])
    
    # Create a DataFrame from the extracted data
    df = pd.DataFrame(data, columns=["Sector", "Description"])
    
    # Print the DataFrame
    print(df)
    
    #Doing some sentiment analysis on each of the new policies 
    
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    nltk.download('vader_lexicon')
    
    sia = SentimentIntensityAnalyzer()
    
    # Perform sentiment analysis and add results to new columns
    df["Sentiment"] = df["Description"].apply(lambda x: sia.polarity_scores(x))
    df["Sentiment_Label"] = df["Sentiment"].apply(lambda x: "Positive" if x["compound"] >= 0.05 else ("Negative" if x["compound"] <= -0.05 else "Neutral"))
    
    # Print the DataFrame with sentiment analysis results
    print(df)
    df.to_csv("Sector_Policies.csv")
    text = ' '.join(df['Description'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    # Display the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for Description Column")
    plt.show()
    
    sector_counts = df['Sector'].value_counts()
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    sector_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Sector')
    plt.ylabel('Count')
    plt.title('Sector Distribution')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.show()



#IEA Data- https://docs.google.com/spreadsheets/d/1K0YalKFSmlQniDXvnqZIsaanVg5Nz14YV7Ptxm5MDP0/edit?usp=sharing
#File ID- 1IpR_1iMI7KPPT7z0reUE8IvIJ1XmEcda

# link= https://drive.google.com/file/d/19iU8WSb8-43KZa37Vnx9g7Yb_Cc-jHMK/view?usp=sharing


def policy_investment_mine():
    url='https://drive.google.com/file/d/16NFG9oi-2QgTHdmPYXniZiAd5j3yuhd5/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df_investment_data = pd.read_csv(url)
    
    
    df_investment_data['Budget commitment (million USD)'] = df_investment_data['Budget commitment (million USD)'].str.replace(' ', '', regex=True).astype(float)
    
    # Print the fixed DataFrame
    print(df_investment_data)
    
    import seaborn as sns
    
    # Category vs Policy Count Heatmap
    heatmap_data = df_investment_data.groupby(['Category', 'Measures']).size().reset_index(name='Count')
    heatmap_pivot = heatmap_data.pivot(index='Category', columns='Measures', values='Count')
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_pivot, cmap='YlGnBu', annot=True, fmt='g')
    plt.title('Category vs Policy Count Heatmap')
    plt.xlabel('Policy Measures')
    plt.ylabel('Category')
    plt.show()
    
    df_investment_data['Start year'] = df_investment_data['Start year'].astype('category')
    
    # Budget Allocation by Category
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_investment_data, x='Category', y='Budget commitment (million USD)', estimator=sum)
    plt.xticks(rotation=45)
    plt.title('Budget Allocation by Category')
    plt.xlabel('Category')
    plt.ylabel('Total Budget (million USD)')
    plt.show()
    
    
    budget_by_category = df_investment_data.groupby('Category')['Budget commitment (million USD)'].sum().reset_index()
    
    # Sort the data by budget in descending order for a better visualization
    budget_by_category = budget_by_category.sort_values(by='Budget commitment (million USD)', ascending=False)
    
    # Create a bar plot
    plt.figure(figsize=(12, 6))
    plt.barh(budget_by_category['Category'], budget_by_category['Budget commitment (million USD)'], color='skyblue')
    plt.xlabel('Budget commitment (million USD)')
    plt.ylabel('Category')
    plt.title('Budget Allocation by Category')
    plt.gca().invert_yaxis()  # Invert y-axis for the largest budget at the top
    plt.show()
    
    text = ' '.join(df_investment_data['Category'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    # Display the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for Description Column")


def price_projections():
    
    
    ####-----------------OPEN PRICE PROJECTIONS BY SECTOR FROM EIA.GOV--------------------------
    
    dataFileUrl = "https://www.eia.gov/outlooks/aeo/excel/aeotab3.xlsx"
    
    ####-----------------READ PRICE PROJECTIONS .XLSX FILE TO DATAFRAME--------------------------
    
    df = pd.read_excel(dataFileUrl, 
                       header=[12])
    
    ####-----------------CLEAN UP HEADERS, COMBINE SECTOR + SOURCE TO A SINGLE LABEL (e.g. Residential: Propane)-------------
    
    df = df.rename(columns={' Sector and Source':'Sector and Source'})
    
    df['Sector and Source'] = df['Sector and Source'].str.strip()
    
    nd_array_sector_source1 = np.array(df.loc[2:5,'Sector and Source'])
    
    nd_array_sector_source1 = 'Residential: ' + nd_array_sector_source1
    
    nd_array_sector_source2 = np.array(df.loc[8:12,'Sector and Source'])
    
    nd_array_sector_source2 = 'Commercial: ' + nd_array_sector_source2
    
    nd_array_sector_source3 = np.array(df.loc[15:22,'Sector and Source'])
    
    nd_array_sector_source3 = 'Industrial: ' + nd_array_sector_source3
    
    nd_array_sector_source4 = np.array(df.loc[25:32,'Sector and Source'])
    
    nd_array_sector_source4 = 'Transportation: ' + nd_array_sector_source4
    
    nd_array_sector_source5 = np.array(df.loc[35:39,'Sector and Source'])
    
    nd_array_sector_source5 = 'Electric Power: ' + nd_array_sector_source5
    
    nd_array_sector_source6 = np.array(df.loc[43:53,'Sector and Source'])
    
    nd_array_sector_source6 = 'Average Price to All Users: ' + nd_array_sector_source6
    
    #####-----------------CREATE % GROWTH ARRAY, not used--------------------------
    
    # avg_annual_change1 = np.array(df.loc[2:5,'2022–2050'])
    
    # avg_annual_change1 = avg_annual_change1.astype(float) * 100
    
    # avg_annual_change1 = np.round(avg_annual_change1,decimals=1)
    
    ####-----------------CREATE DATAFRAME/PLOT FOR RESIDENTIAL PRICES--------------------------
    
    values_sector_1 = np.array([df.loc[2:5,2023],df.loc[2:5,2030],df.loc[2:5,2040],df.loc[2:5,2050]])
    
    values_sector_1 = values_sector_1.astype(float)
    
    values_sector_1 = np.round(values_sector_1,decimals=2)                      
    
    residential_table = pd.DataFrame(values_sector_1,columns=nd_array_sector_source1,index=['2023','2030','2040','2050'])
    
    plt.style.use('ggplot')
    
    plot1 = residential_table.plot(rot=45,kind='line')
    
    ####-----------------CREATE DATAFRAME/PLOT FOR COMMERCIAL PRICES--------------------------
    
    values_sector_2 = np.array([df.loc[8:12,2023],df.loc[8:12,2030],df.loc[8:12,2040],df.loc[8:12,2050]])
    
    values_sector_2 = values_sector_2.astype(float)
    
    values_sector_2 = np.round(values_sector_2,decimals=2)
    
    commercial_table = pd.DataFrame(values_sector_2,columns=nd_array_sector_source2,index=['2023','2030','2040','2050'])
    
    plt.style.use('ggplot')
    
    plot2 = commercial_table.plot(rot=45,kind='line')
    
    ####-----------------CREATE DATAFRAME/PLOT FOR INDUSTRIAL PRICES--------------------------
    
    
    values_sector_3 = np.array([df.loc[15:22,2023],df.loc[15:22,2030],df.loc[15:22,2040],df.loc[15:22,2050]])
    
    values_sector_3[:,6] = 0.00
    
    values_sector_3 = values_sector_3.astype(float)
    
    values_sector_3 = np.round(values_sector_3,decimals=2)
    
    industrial_table = pd.DataFrame(values_sector_3,columns=nd_array_sector_source3,index=['2023','2030','2040','2050'])
    
    plt.style.use('ggplot')
    
    plot3 = industrial_table.plot(rot=45,kind='line')
    
    ####-----------------CREATE DATAFRAME/PLOT FOR TRANSPORTATION PRICES--------------------------
    
    
    values_sector_4 = np.array([df.loc[25:32,2023],df.loc[25:32,2030],df.loc[25:32,2040],df.loc[25:32,2050]])
    
    values_sector_4 = values_sector_4.astype(float)
    
    values_sector_4 = np.round(values_sector_4,decimals=2)
    
    transportation_table = pd.DataFrame(values_sector_4,columns=nd_array_sector_source4,index=['2023','2030','2040','2050'])
    
    plt.style.use('ggplot')
    
    plot4 = transportation_table.plot(rot=45,kind='line')
    
    ####-----------------CREATE DATAFRAME/PLOT FOR ELECTRIC POWER PRICES--------------------------
    
    values_sector_5 = np.array([df.loc[35:39,2023],df.loc[35:39,2030],df.loc[35:39,2040],df.loc[35:39,2050]])
    
    values_sector_5 = values_sector_5.astype(float)
    
    values_sector_5 = np.round(values_sector_5,decimals=2)
    
    electric_power_table = pd.DataFrame(values_sector_5,columns=nd_array_sector_source5,index=['2023','2030','2040','2050'])
    
    
    plt.style.use('ggplot')
    
    plot5 = electric_power_table.plot(rot=45,kind='line')
    
    
    ####-----------------PRINT ALL 5 DATA FRAMES AND CORRESPONDING PLOTS--------------------------
    
    print(residential_table,plot1)
    
    print(commercial_table,plot2)
    
    print(industrial_table,plot3)
    
    print(transportation_table,plot4)
    
    print(electric_power_table,plot5)
    
    
    ####-----------------COMBINE ALL 5 DATA FRAMES--------------------------------------------------
    
    
    price_projections = [residential_table,
                         commercial_table,
                         industrial_table,
                         transportation_table,
                         electric_power_table]
    
    
    price_projections = pd.concat(price_projections,axis=1)
    
    
    ####-----------------PRINT ALL PRICE PROJECTION DATA---------------------------------------------
    
    print(price_projections)

print("---Welcome to Green Investment Strategies, what do you want to know today?---")
print("What do you want to know today\n1.US Energy Policy Analysis and Updates \n2.Investment Analysis for Policy \n3.Energy Price Projections")
print("Enter your prompt")
try: 
    a=int(input())
    if a == 1:
        policy_mine()
    elif a==2: 
        policy_investment_mine()
    elif a==3: 
        price_projections()
    else: 
        print("Wrong Input Detected, try again")
except: 
    print("Wrong Input")
        