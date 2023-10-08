#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import requests
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
from io import BytesIO

#In order to instlal any dependencies not available in your system use the requirements.txt file 
#pip install -r requirements.txt

def policy_mine():
    
    # Send a GET request to the webpage
    url = "https://www.whitehouse.gov/climate/"
    response = requests.get(url)
    
    
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
    else: 
        print("Error making request, try after some time")
    
    #We extract the particular element that stores the policy data we are looking for
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

    nltk.download('vader_lexicon')
    
    sia = SentimentIntensityAnalyzer()
    
    # Perform sentiment analysis and add results to new columns
    df["Sentiment"] = df["Description"].apply(lambda x: sia.polarity_scores(x))
    df["Sentiment_Label"] = df["Sentiment"].apply(lambda x: "Positive" if x["compound"] >= 0.05 else ("Negative" if x["compound"] <= -0.05 else "Neutral"))
    
    # Separate data into positive and negative sentiments
    positive_data = df[df['Sentiment_Label'] == 'Positive']
    negative_data = df[df['Sentiment_Label'] == 'Negative']
    
        
    sentiment_counts = df['Sentiment_Label'].value_counts()

    # Define colors
    colors = ['#66b3ff', '#ff9999']
    
    # Create a pie chart to understand the sentiment of the policy text that we have scraped
    plt.figure(figsize=(6, 6))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'black'}, 
        textprops={'fontsize': 12},
    )
    plt.title('Sentiment Distribution- How are the latest policies perceived by the world', fontsize=16)
    
    # Display the chart
    plt.show()
    
    
    
    #Writing to the dataframe if required
    #df.to_csv("Sector_Policies.csv")
    
    #Plotting a word cloud for the given data
    text = ' '.join(df['Description'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    
    print("\nDisplaying the emerging trends in the energy sector\n")
    # Display the WordCloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for Description Column")
    plt.show()
    
    
    print("\nDisplaying the count of policies by sector\n")
    #Counting the policy by sectors and plotting a bar chart for the same
    sector_counts = df['Sector'].value_counts()
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    sector_counts.plot(kind='bar', color='skyblue')
    plt.xlabel('Sector')
    plt.ylabel('Count')
    plt.title('Sector Distribution')
    plt.show()




def policy_investment_mine():
    
    #Sending a request to the database for IEA Data
    url='https://drive.google.com/file/d/16NFG9oi-2QgTHdmPYXniZiAd5j3yuhd5/view?usp=sharing'
    url='https://drive.google.com/uc?id=' + url.split('/')[-2]
    df_investment_data = pd.read_csv(url)
    
    
    df_investment_data['Budget commitment (million USD)'] = df_investment_data['Budget commitment (million USD)'].str.replace(' ', '', regex=True).astype(float)
    
    # Print the fixed DataFrame
    print(df_investment_data)
    

    
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
    
    
    
    #Cleaning the data by removing the empty rows and backfilling it with appropriate values
    df_investment_data['Policy'].fillna(method='ffill', inplace=True)
    df_investment_data = df_investment_data[df_investment_data['Category'] != 'Multiple']
    
    
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

def emissions_by_year_and_fuel() :    
    #Original URL
    url = 'https://www.eia.gov/environment/emissions/state/'
    
    # Fetching the content of EIA site
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the the EIA site to look for excel on page
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Storing all Excel file links ending in .xls or .xlsx
    excel_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith(('.xls', '.xlsx'))][:2]
    
    dataframes = []
    
    
    # Code for file 1 (Table1):
        
        #Preparing data for Table 1
    
    response = requests.get(url + excel_links[0])
    response.raise_for_status()
        
    # Reading the Excel file into a pandas DataFrame with multi-row headers
    
    excel_file = BytesIO(response.content)
    
    #Actual data starts with 4 row on the excel
    
    df = pd.read_excel(excel_file, skiprows=4, engine='openpyxl')  # 'openpyxl' is used to read .xlsx files
    
    columns = df.columns.tolist()
    states = df['State']
    #print(states)
    year=df.columns[1:53:8]
    #print(year)
    
    # Modifying the last 4 columns to get an effective change plot
    columns[-4] = "Change (1970-2021) Percent"
    columns[-3] = "Change (1970-2021) Absolute"
    columns[-2] = "Change (2019-2021) Percent"
    columns[-1] = "Change (2019-2021) Absolute"
    
    # Flattening the multi-level columns and joining them to make better sense
    df.columns = columns
    df = df.iloc[:-2]
    #print(df[0:2])
    #print(df.head())
    
    #PLOTTING THE CHANGE ACROSS States for 2019-2020 and 1970-2021
    
   #Plottting the change against states for 2 periods, to compare change in 1 year and change in 50 years 
    states = df['State']
    print(states)
    change_1970_2021_percent = df['Change (1970-2021) Percent']
    change_2019_2021_percent = df['Change (2019-2021) Percent']
    # Plotting the data
    fig, ax = plt.subplots(figsize=(40, 10))
    bar_width = 0.5
    
    # Bar positions
    index = range(len(states))
    
    # Plotting the bars
    rects1 = ax.bar(index, change_1970_2021_percent, bar_width, label='Change (1970-2021) Percent')
    rects2 = ax.bar([p + bar_width for p in index], change_2019_2021_percent, bar_width, label='Change (2019-2021) Percent')
    
    # Adding labels, title, and custom x-axis tick labels
    ax.set_xlabel('States')
    ax.set_ylabel('Change Percentage')
    ax.set_title('Change in Emissions Percentage (1970-2021 and 2019-2021)')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(states, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    
    #Code to read FILE 2 (TABLE 2) :
        # Same procedure for table 2
        #GETTING TABLE 2 from the excel links created above
    response = requests.get(url + excel_links[1])
    response.raise_for_status()
    
    excel_file = BytesIO(response.content)
    df_table2 = pd.read_excel(excel_file, skiprows=3, engine='openpyxl')  # 'openpyxl' is used to read .xlsx files
    
    columns = df_table2.columns.tolist()
    #print(columns)
    #Eliminating the last 2 columns in table 2 as they read some informative text
    states = df_table2['State'][:-2]
    #print(states)
    #Last 3 columns in excel give % of emissions by Coal, Petroleum and Natural Gas
    coal_percentages = df_table2['Coal.1'][:-2]
    natural_gas_percentages = df_table2['Petroleum.1'][:-2]
    petroleum_percentages = df_table2['Natural Gas'][:-2]
    #print(states)
    
    #Plotting the bar chart 
    fig, ax = plt.subplots(figsize=(10,8))
    bar_width = 0.25
    index = states.tolist()
    
    # The positions for the bars
    position = np.arange(len(index))
    
    # Plotting the bars
    bar1 = ax.barh(position, coal_percentages, color='blue', label='Coal')
    bar2 = ax.barh(position, natural_gas_percentages, left=coal_percentages, color='green', label='Natural Gas')
    bar3 = ax.barh(position, petroleum_percentages, left=coal_percentages + natural_gas_percentages, color='red', label='Petroleum')
    
    # Set y-labels to be the states
    ax.set_yticks(position)
    ax.set_yticklabels(states)
    
    # Set x-labels to be percentage emissions
    ax.set_xlabel('Percentage of CO2 Emissions')
    
    ax.set_title('Percentage split of CO2 Emissions by state')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def emissions_by_sector_state():


    # Accessing the dataset 
    # URL of the Excel file
    excel_url = "https://www.eia.gov/environment/emissions/state/excel/table3.xlsx"
    
    # Read the Excel file into a Pandas DataFrame, skipping rows 1 to 3 (header) 
    df = pd.read_excel(excel_url, skiprows=range(1, 3), header = 0)

#-----------------------------------------------------------------------------------------------------------------------
    
    # Data cleaning
    
    #Dropping unwanted rows
    df = df.drop(df.index[53:])
    
    #creating a header
    header_row = df.iloc[0]
    df = df[1:]
    df.columns = header_row
    
    #Extracting the total column from the dataset
    last_row = df.iloc[-1:]  # Select the last row
    state_sector_totals = pd.DataFrame(last_row)
    state_Sector_df = df.drop(df.index[-1])
    
    state_Sector_df = state_Sector_df.iloc[:, :7].copy()
    
#-----------------------------------------------------------------------------------------------------------------------
    
    # Vizualization
    # 1 Donut Chart
    # Select the relevant columns and convert decimals to percentages
    percentages = df.iloc[:, -5:] * 100
    
    # Data for the donut chart
    labels = percentages.columns
    sizes = percentages.iloc[0].tolist()
    
    # Create a donut chart
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    
    # Draw a white circle at the center to create a donut effect
    center_circle = plt.Circle((0,0),0.70,fc='white')
    fig.gca().add_artist(center_circle)
    ax.axis('equal')  
    
    # Title
    plt.title('Sector wise emissions in the US - 2021', y=1.08)
    
    # Show the plot
    plt.show()
 
 #-----------------------------------------------------------------------------------------------------------------------
    
    # 2 Bar chart
    
    # Sort the DataFrame by Total emissions in descending order and select the top 10
    top_10_states = state_Sector_df.sort_values(by="Total", ascending=False).head(15)
    
    # Create a color palette (you can customize the colors as needed)
    colors = sns.color_palette("viridis", len(top_10_states))
    
    # Create the bar graph with colors
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_10_states["State"], top_10_states["Total"], color=colors)
    
    # Add data labels (rounded to 1 decimal place) above the bars
    for bar, emission in zip(bars, top_10_states["Total"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2 - 0.1,
            emission + 5,
            f"{emission:.1f}",  # Format to 1 decimal place
            fontsize=10,
        )
    
    plt.title("Top 10 States by Emissions")
    plt.xlabel("State")
    plt.ylabel("Total Emissions (Million Metrics tons of CO2)")
    plt.xticks(rotation=45, ha="right")  # Rotate the state names for better readability
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    
#-----------------------------------------------------------------------------------------------------------------------    
    #3 Bar chart for comparision
    # Sort the DataFrame by Total emissions in descending order and select the top 10
    top_15_states = state_Sector_df.sort_values(by="Total", ascending=False).head(15)
    
    # Calculate the total emissions for the top 10 states
    top_15_total_emissions = top_15_states["Total"].sum()
    
    # Calculate the total emissions for all states
    total_emissions = state_Sector_df["Total"].sum()
    
    # Calculate the percentage share for the top 10 states and the remaining states
    top_15_percentage = (top_15_total_emissions / total_emissions) * 100
    remaining_percentage = 100 - top_15_percentage
    
    # Create a DataFrame for the percentages
    percentage_df = pd.DataFrame({
        "Category": ["Top 15 States", "Remaining States"],
        "Percentage": [top_15_percentage, remaining_percentage]
    })
    
    # Create a bar graph
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Category", y="Percentage", data=percentage_df, palette="Set2")
    plt.title("Percentage Share of Emissions")
    plt.ylabel("Percentage")
    plt.ylim(0, 100)  # Set the y-axis range to 0-100
    
    # Add data labels above the bars
    for index, row in percentage_df.iterrows():
        plt.text(index, row["Percentage"] + 2, f"{row['Percentage']:.1f}%", ha="center", fontsize=12)
    
    plt.tight_layout()
    
    # Display the plot
    plt.show()
 
#-----------------------------------------------------------------------------------------------------------------------
    
    # Data cleaning for further viz
    top_15_states = top_15_states.drop(columns=['Total'])
   
    # 4 stacked bar chart
    
    ax = top_15_states.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Emissions by State - Top 15 - 2021')
    plt.xlabel('States')
    plt.ylabel('Emissions (Million Metrics tons of CO2)')
    plt.legend(title='Sectors', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------------------------------------------    
    # Data cleaning for further viz
    heatmap_df = state_Sector_df
    # Set the "State" column as the index
    heatmap_df.set_index('State', inplace=True)
    # Exclude the "Total" column from the DataFrame
    heatmap_df = heatmap_df.drop(columns=['Total'])
    # Convert all columns to numeric (excluding 'State' column)
    heatmap_df = heatmap_df.apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with 0 (or any other suitable value)
    heatmap_df.fillna(0, inplace=True)
#-----------------------------------------------------------------------------------------------------------------------    
    #5 Heatmap 
    plt.figure(figsize=(12,30))
    sns.heatmap(heatmap_df, cmap='YlGnBu', annot=True, fmt=".1f", cbar_kws={'label': 'Emissions'})
    plt.title('Emissions Across States and Sectors - 2021')
    plt.xlabel('Sectors')
    plt.ylabel('States')
    plt.tight_layout()
    plt.show()
#-----------------------------


#The menu driven program, mines data from 5 data sources for the customer's analysis
print("\n---Welcome to Green Investment Strategies, what do you want to know today?---")
print()
print("What do you want to know today\n1.US Energy Policy Analysis and Updates \n2.Investment Analysis for Policy \n3.Energy Price Projections\n4.Emissions by Year and Fuel\n5.Emissions by Sector and State")
print("Enter your prompt")
try: 
    a=int(input())
    if a == 1:
        policy_mine()
    elif a==2: 
        policy_investment_mine()
    elif a==3: 
        price_projections()
    elif a==4:
        emissions_by_year_and_fuel()
    elif a==5:
        emissions_by_sector_state()
    else: 
        print("Wrong Input Detected, try again")
except: 
    print("Wrong Input")
        