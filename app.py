import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import sys
import ast
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the data
df = pd.read_csv("Fashion.csv")

# Function to categorize satisfaction status
def satisfaction_score(rating):
    if rating >= 4.0:
        return 'Satisfied'
    elif rating >= 2.5:
        return 'Neutral'
    else:
        return 'Dissatisfied'

# Apply satisfaction score function
df['Satisfaction_Status'] = df['Review Rating'].apply(satisfaction_score)

# Function to map items to categories
mapping = {
    'Belt': 'Accessory', 'Handbag': 'Accessory', 'Blouse': 'Top', 'Polo Shirt': 'Top', 'Raincoat': 'Outerwear', 'Tank Top': 'Top', 'Flip-Flops': 'Footwear', 'Hoodie': 'Outerwear',
    'Socks': 'Footwear', 'Bowtie': 'Accessory', 'Sunglasses': 'Accessory', 'T-shirt': 'Top', 'Poncho': 'Outerwear', 'Vest': 'Outerwear',
    'Skirt': 'Bottom',
    'Blazer': 'Outerwear',
    'Pajamas': 'Sleepwear',
    'Jacket': 'Outerwear',
    'Gloves': 'Accessory',
    'Slippers': 'Footwear',
    'Backpack': 'Accessory',
    'Boots': 'Footwear',
    'Sun Hat': 'Accessory',
    'Trousers': 'Bottom',
    'Overalls': 'Bottom',
    'Camisole': 'Top',
    'Trench Coat': 'Outerwear',
    'Pants': 'Bottom',
    'Romper': 'One-piece',
    'Sneakers': 'Footwear',
    'Jumpsuit': 'One-piece',
    'Onesie': 'One-piece',
    'Flannel Shirt': 'Top',
    'Coat': 'Outerwear',
    'Leggings': 'Bottom',
    'Sandals': 'Footwear',
    'Wallet': 'Accessory',
    'Tie': 'Accessory',
    'Shorts': 'Bottom',
    'Loafers': 'Footwear',
    'Kimono': 'Outerwear',
    'Scarf': 'Accessory',
    'Cardigan': 'Outerwear',
    'Hat': 'Accessory',
    'Swimsuit': 'Swimwear',
    'Tunic': 'Top',
    'Dress': 'One-piece',
    'Sweater': 'Top',
    'Umbrella': 'Accessory',
    'Jeans': 'Bottom'
}

df['Item Purchased'] = df['Item Purchased'].replace(mapping)

# Convert 'Date Purchase' to datetime format
df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])

# Shuffle the DataFrame to demonstrate changes
df = df.sample(frac=1).reset_index(drop=True)

# Add a new column 'state' with alternating values 'a' and 'b'
df['state'] = ['a' if i < len(df) / 2 else 'b' for i in range(len(df))]

# Insert the new column at the beginning of the DataFrame
df.insert(0, 'state', df.pop('state'))

# Ensure the data is sorted by date
df = df.sort_values(by='Date Purchase')


# Function to create Plotly table
def create_plotly_table(state_df, state_name):
    value_counts = state_df['Item Purchased'].value_counts()
    
    table = go.Table(
        header=dict(values=[f"Category (State {state_name})", "Count"],
                    fill_color='black',
                    font=dict(color='white')),
        cells=dict(values=[value_counts.index, value_counts.values],
                   fill_color='white',
                   font=dict(color='black')),
        columnwidth=[120, 80],
        header_align=['center', 'center'],
        cells_align=['center', 'center']
    )

    layout = go.Layout(
        width=800,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    fig = go.Figure(data=[table], layout=layout)
    st.plotly_chart(fig)


# Function to create seaborn bar plot
def create_seaborn_barplot(df):
    top_purchased_products = df.groupby(['state', 'Item Purchased']).size().reset_index(name='Count')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Item Purchased', y='Count', hue='state', data=top_purchased_products)
    plt.title('Top Purchased Products by State for all months')
    plt.xlabel('Item Purchased')
    plt.ylabel('Number of Purchases')
    plt.xticks(rotation=45)
    plt.legend(title='State')
    st.pyplot(plt)


# Function to show items purchased by each state and its visual representation
def show_items_purchased_by_state():
    st.subheader("Items Purchased by Each State and Its Visual Representation")    
    # Ensure 'state' column is available
    if 'state' not in df.columns:
        st.error("The 'state' column is missing in the DataFrame. Please check your data loading and manipulation steps.")
        return
    
    # Get unique states
    states = df['state'].unique()
    
    # Select state using dropdown
    selected_state = st.selectbox("Select State", states)
    
    # Filter data for selected state
    df_state = df[df['state'] == selected_state]
    
    # Get value counts for each item purchased
    value_counts = df_state['Item Purchased'].value_counts()
    
    # Create Plotly table
    table = go.Table(
        header=dict(values=[f"Category (State {selected_state})", "Count"],
                    fill_color='black',
                    font=dict(color='white')),
        cells=dict(values=[value_counts.index, value_counts.values],
                   fill_color='white',
                   font=dict(color='black')),
        columnwidth=[120, 80],
        header_align=['center', 'center'],
        cells_align=['center', 'center']
    )

    layout = go.Layout(
        width=800,
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    fig = go.Figure(data=[table], layout=layout)
    
    # Display the Plotly table using st.plotly_chart
    st.plotly_chart(fig)

    # Create seaborn bar plot for top purchased products
    top_purchased_products = df_state.groupby(['Item Purchased']).size().reset_index(name='Count')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Item Purchased', y='Count', data=top_purchased_products)
    plt.title(f'Top Purchased Products in State {selected_state}')
    plt.xlabel('Item Purchased')
    plt.ylabel('Number of Purchases')
    plt.xticks(rotation=45)
    st.pyplot(plt)


# Function to show monthly purchases
def show_monthly_purchases():
    st.subheader("Monthly Purchases by State")
    
    # Ensure 'state' column is available
    if 'state' not in df.columns:
        st.error("The 'state' column is missing in the DataFrame. Please check your data loading and manipulation steps.")
        return
    
    # Calculate monthly purchases
    df['Month'] = df['Date Purchase'].dt.to_period('M')
    monthly_purchases = df.groupby(['Month', 'state']).size().reset_index(name='Count')
    
    # Plot using seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='Count', hue='state', data=monthly_purchases)
    plt.title('Monthly Purchases by State for all products')
    plt.xlabel('Month')
    plt.ylabel('Number of Purchases')
    plt.xticks(rotation=45)
    plt.legend(title='State')
    
    # Display plot in Streamlit
    st.pyplot(plt)


# Function to show payment methods by each state
def show_payment_method_by_state():
    st.subheader("Payment Method by Each State")
    
    # Ensure 'state' column is available
    if 'state' not in df.columns:
        st.error("The 'state' column is missing in the DataFrame. Please check your data loading and manipulation steps.")
        return
    
    # Separate data by state
    df_a = df[df['state'] == 'a']
    df_b = df[df['state'] == 'b']
    
    # Plotting side by side using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plotting for State A
    counts_a = df_a["Payment Method"].value_counts()
    explode = (0, 0.1)  # Explode the 2nd slice (if desired)
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange colors
    
    counts_a.plot(kind='pie', fontsize=12, colors=colors, explode=explode, autopct='%1.1f%%', ax=axes[0])
    axes[0].set_xlabel('Payment Method - State A', weight="bold", color="#ff2400", fontsize=15, labelpad=20)
    axes[0].axis('equal')
    axes[0].legend(labels=counts_a.index, loc="best")
    
    # Plotting for State B
    counts_b = df_b["Payment Method"].value_counts()
    
    counts_b.plot(kind='pie', fontsize=12, colors=colors, explode=explode, autopct='%1.1f%%', ax=axes[1])
    axes[1].set_xlabel('Payment Method - State B', weight="bold", color="#ff2400", fontsize=15, labelpad=20)
    axes[1].axis('equal')
    axes[1].legend(labels=counts_b.index, loc="best")
    
    # Display the plot in Streamlit
    st.pyplot(fig)


# Function to perform customer segment analysis by states
def customer_segment_analysis():
    st.subheader("Customer Segment Analysis by States")
    
    # Ensure 'state' column is available
    if 'state' not in df.columns:
        st.error("The 'state' column is missing in the DataFrame. Please check your data loading and manipulation steps.")
        return
    
    # Define the customer segments based on satisfaction levels
    df['Satisfaction_Score'] = pd.cut(df['Review Rating'], bins=[0, 2.49, 3.99, 5], labels=['Dissatisfied', 'Neutral', 'Satisfied'])
    
    # Analyze the characteristics of each segment for each state
    segment_analysis_a = df[df['state'] == 'a'].groupby('Satisfaction_Score')['Item Purchased'].count().reset_index()
    segment_analysis_b = df[df['state'] == 'b'].groupby('Satisfaction_Score')['Item Purchased'].count().reset_index()
    
    # Print statements for debugging
    print("Segment Analysis A:")
    print(segment_analysis_a)
    print("Segment Analysis B:")
    print(segment_analysis_b)
    
    # Create Plotly tables for each state
    table_a = go.Figure(data=[go.Table(
        header=dict(values=['Satisfaction Score', 'Count of Items Purchased'],
                    fill_color='lightgrey',
                    align='left',
                    font=dict(color='black')),
        cells=dict(values=[segment_analysis_a['Satisfaction_Score'],
                           segment_analysis_a['Item Purchased']],
                   fill_color='white',
                   align='left',
                   font=dict(color='black')))
    ])
    
    table_b = go.Figure(data=[go.Table(
        header=dict(values=['Satisfaction Score', 'Count of Items Purchased'],
                    fill_color='lightgrey',
                    align='left',
                    font=dict(color='black')),
        cells=dict(values=[segment_analysis_b['Satisfaction_Score'],
                           segment_analysis_b['Item Purchased']],
                   fill_color='white',
                   align='left',
                   font=dict(color='black')))
    ])
    
    # Print statements for debugging
    print("Table A Data:")
    print(table_a)
    print("Table B Data:")
    print(table_b)
    
    # Update layout and titles for each table
    table_a.update_layout(title='Customer Segment Analysis (State A)',
                          title_x=0.5,
                          autosize=False)
    
    table_b.update_layout(title='Customer Segment Analysis (State B)',
                          title_x=0.5,
                          autosize=False)
    
    # Display the tables using st.plotly_chart
    st.plotly_chart(table_a)
    st.plotly_chart(table_b)



# Function to plot side-by-side bar graphs with different colors
def plot_side_by_side_bars(df1, df2, label1, label2):
    plt.figure(figsize=(12, 6))

    # Define colors for bars
    colors_a = ['#1f77b4', '#ff7f0e', 'salmon']
    colors_b = ['#1f77b4', '#ff7f0e', 'salmon']

    # Plot for State A
    plt.subplot(1, 2, 1)
    plt.bar(df1['Satisfaction_Score'], df1['Item Purchased'], color=colors_a)
    plt.xlabel('Satisfaction Score', weight='bold', fontsize=12)
    plt.ylabel('Count of Items Purchased', weight='bold', fontsize=12)
    plt.title(f'Satisfaction Score Distribution - {label1}', weight='bold', fontsize=14)
    plt.xticks(rotation=45)

    # Plot for State B
    plt.subplot(1, 2, 2)
    plt.bar(df2['Satisfaction_Score'], df2['Item Purchased'], color=colors_b)
    plt.xlabel('Satisfaction Score', weight='bold', fontsize=12)
    plt.ylabel('Count of Items Purchased', weight='bold', fontsize=12)
    plt.title(f'Satisfaction Score Distribution - {label2}', weight='bold', fontsize=14)
    plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot(plt.gcf())  # Display the plot using Streamlit's pyplot integration

# Function to display satisfaction scores
def display_satisfaction_scores():
    st.subheader("Satisfaction Scores")
    
    # Define the customer segments based on satisfaction levels
    df['Satisfaction_Score'] = pd.cut(df['Review Rating'], bins=[0, 2.49, 3.99, 5], labels=['Dissatisfied', 'Neutral', 'Satisfied'])
    
    # Analyze the characteristics of each segment for each state
    segment_analysis_a = df[df['state'] == 'a'].groupby('Satisfaction_Score')['Item Purchased'].count().reset_index()
    segment_analysis_b = df[df['state'] == 'b'].groupby('Satisfaction_Score')['Item Purchased'].count().reset_index()
    
    # Plot side-by-side bars for States A and B
    plot_side_by_side_bars(segment_analysis_a, segment_analysis_b, 'State A', 'State B')



# Function to display top 10 customers by purchase amount for each state
def top_10_customers():
    st.subheader("Top 10 Customers by Purchase Amount")
    
    # Calculate total purchase amount for each customer in each state
    customer_totals_a = df[df['state'] == 'a'].groupby('Customer Reference ID')['Purchase Amount (USD)'].sum().reset_index()
    customer_totals_b = df[df['state'] == 'b'].groupby('Customer Reference ID')['Purchase Amount (USD)'].sum().reset_index()

    # Sort the customers by their total purchase amount in descending order and get the top 10 for each state
    top_10_customers_a = customer_totals_a.nlargest(10, 'Purchase Amount (USD)')
    top_10_customers_b = customer_totals_b.nlargest(10, 'Purchase Amount (USD)')

    # Create Plotly figures with tables for each state
    fig_a = go.Figure(data=[go.Table(
        header=dict(values=['Customer Reference ID (State A)', 'Purchase Amount (USD)'],
                    fill_color='turquoise',
                    align='left',
                    font=dict(color='black')),
        cells=dict(values=[top_10_customers_a['Customer Reference ID'], top_10_customers_a['Purchase Amount (USD)']],
                   fill_color='lavender',
                   align='left',
                   font=dict(color='black')))
    ])

    fig_b = go.Figure(data=[go.Table(
        header=dict(values=['Customer Reference ID (State B)', 'Purchase Amount (USD)'],
                    fill_color='turquoise',
                    align='left',
                    font=dict(color='black')),
        cells=dict(values=[top_10_customers_b['Customer Reference ID'], top_10_customers_b['Purchase Amount (USD)']],
                   fill_color='lavender',
                   align='left',
                   font=dict(color='black')))
    ])

    # Update the layout of the tables
    fig_a.update_layout(title='Top 10 Customers by Purchase Amount (State A)')
    fig_b.update_layout(title='Top 10 Customers by Purchase Amount (State B)')

    # Display the tables using st.plotly_chart
    st.plotly_chart(fig_a)
    st.plotly_chart(fig_b)



# Function to display total profit comparison for 2022 and 2023 by states
def total_profit_comparison():
    st.subheader("Total Profit Comparison for 2022 and 2023 by States")
    
    # Filter the dataset for the year 2022 and 2023 for each state
    df_2022_a = df[(df['Date Purchase'].dt.year == 2022) & (df['state'] == 'a')]
    df_2023_a = df[(df['Date Purchase'].dt.year == 2023) & (df['state'] == 'a')]

    df_2022_b = df[(df['Date Purchase'].dt.year == 2022) & (df['state'] == 'b')]
    df_2023_b = df[(df['Date Purchase'].dt.year == 2023) & (df['state'] == 'b')]

    # Calculate the total profit for 2022 and 2023 for each state
    total_profit_2022_a = df_2022_a['Purchase Amount (USD)'].sum()
    total_profit_2023_a = df_2023_a['Purchase Amount (USD)'].sum()

    total_profit_2022_b = df_2022_b['Purchase Amount (USD)'].sum()
    total_profit_2023_b = df_2023_b['Purchase Amount (USD)'].sum()

    # Create data for the table
    header = ['State', 'Year', 'Total Profit']
    data = [
        ['State A', '2022', total_profit_2022_a],
        ['State A', '2023', total_profit_2023_a],
        ['State B', '2022', total_profit_2022_b],
        ['State B', '2023', total_profit_2023_b]
    ]

    # Create a Plotly table
    table = go.Figure(data=[go.Table(
        header=dict(values=header, fill=dict(color='black'), font=dict(color='white')),
        cells=dict(values=list(zip(*data)), fill=dict(color='grey'), font=dict(color='black'))
    )])

    # Update the layout for the table
    table.update_layout(
        title='Total Profit Comparison for 2022 and 2023 by State',
        width=800,
        height=300,
        paper_bgcolor='white',
        font=dict(color='black')
    )

    # Display the Plotly table using st.plotly_chart
    st.plotly_chart(table)

    # # Create data for the bar chart
    # states = ['State A', 'State A', 'State B', 'State B']
    # years = ['2022', '2023', '2022', '2023']
    # total_profits = [total_profit_2022_a, total_profit_2023_a, total_profit_2022_b, total_profit_2023_b]

    # # Create a Plotly bar chart
    # bar_chart = go.Figure(data=[
    #     go.Bar(name='2022', x=states, y=[total_profit_2022_a, total_profit_2022_b]),
    #     go.Bar(name='2023', x=states, y=[total_profit_2023_a, total_profit_2023_b])
    # ])

    # # Update layout for the bar chart
    # bar_chart.update_layout(
    #     title='Total Profit Comparison for 2022 and 2023 by State',
    #     xaxis_title='State',
    #     yaxis_title='Total Profit (USD)',
    #     barmode='group',
    #     width=800,
    #     height=400,
    #     paper_bgcolor='white',
    #     font=dict(color='black')
    # )

    # # Display the Plotly bar chart using st.plotly_chart
    # st.plotly_chart(bar_chart)
     # Create data for the bar chart
    states = ['State A', 'State A', 'State B', 'State B']
    years = ['2022', '2023', '2022', '2023']
    total_profits = [total_profit_2022_a, total_profit_2023_a, total_profit_2022_b, total_profit_2023_b]

    # Create a Plotly bar chart
    fig_bar = go.Figure(data=[
        go.Bar(name='2022', x=['State A', 'State B'], y=[total_profit_2022_a, total_profit_2022_b]),
        go.Bar(name='2023', x=['State A', 'State B'], y=[total_profit_2023_a, total_profit_2023_b])
    ])

    # Update layout for bar chart
    fig_bar.update_layout(
        title='Total Profit Comparison for 2022 and 2023 by State',
        xaxis_title='State',
        yaxis_title='Total Profit (USD)',
        barmode='group',
        width=600,
        height=400,
        paper_bgcolor='white'
    )

    # Create the data for the table
    header = ['State', 'Year', 'Total Profit']
    data = [
        ['State A', '2022', total_profit_2022_a],
        ['State A', '2023', total_profit_2023_a],
        ['State B', '2022', total_profit_2022_b],
        ['State B', '2023', total_profit_2023_b]
    ]

    # Create a Plotly table
    fig_table = go.Figure(data=[go.Table(
        header=dict(values=header, fill=dict(color='black'), font=dict(color='white')),
        cells=dict(values=list(zip(*data)), fill=dict(color='grey'), font=dict(color='white'))
    )])

    # Update layout for table
    fig_table.update_layout(
        title='Total Profit Comparison for 2022 and 2023 by State',
        width=600,
        height=350,
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Display both the table and the bar chart using Streamlit
    st.plotly_chart(fig_bar)


# Function to display top 5 items purchased by month in each state
def get_top_items_by_month(state_data):
    state_data['Month'] = pd.to_datetime(state_data['Date Purchase']).dt.to_period('M')
    grouped = state_data.groupby(['Month', 'Item Purchased']).size().reset_index(name='Count')
    top_items = grouped.sort_values(['Month', 'Count'], ascending=[True, False]).groupby('Month').head(5)
    return top_items

# Separate data by state
state_a_data = df[df['state'] == 'a']
state_b_data = df[df['state'] == 'b']

# Get top items for each state
top_items_a = get_top_items_by_month(state_a_data)
top_items_b = get_top_items_by_month(state_b_data)

# Function to format top items for display with dynamic background colors for months
def format_top_items(top_items, state_name):
    formatted_output = f"<div class='state-section'>" \
                       f"<div class='state-header'>Top Items Purchased for State {state_name.upper()}</div>" \
                       f"<div class='item-list'>" \
                       f"<table>" \
                       f"<thead>" \
                       f"<tr>" \
                       f"<th>Month</th>" \
                       f"<th>Item Purchased</th>" \
                       f"<th>Count</th>" \
                       f"</tr>" \
                       f"</thead>" \
                       f"<tbody>"

    # Define colors for months (example colors, replace with your desired colors)
    month_colors = {
        '2022-10': '#007bff',  # Blue
        '2022-11': '#6f42c1',  # Purple
        '2022-12': '#28a745',  # Green
        '2023-01': '#dc3545',  # Red
        '2023-02': '#fd7e14',  # Orange
        '2023-03': '#6610f2',  # Indigo
        '2023-04': '#17a2b8',  # Cyan
        '2023-05': '#ffc107',  # Yellow
        '2023-06': '#e83e8c',  # Pink
        '2023-07': '#20c997',  # Teal
        '2023-08': '#6c757d',  # Gray
        '2023-09': '#343a40'   # Dark Gray
    }

    for index, row in top_items.iterrows():
        month_color = month_colors.get(str(row['Month']), '#343a40')  # Default to dark gray if month not in dictionary
        formatted_output += f"<tr style='background-color: {month_color};'>" \
                            f"<td>{row['Month']}</td>" \
                            f"<td>{row['Item Purchased']}</td>" \
                            f"<td>{row['Count']}</td>" \
                            f"</tr>"

    formatted_output += f"</tbody>" \
                       f"</table>" \
                       f"</div>" \
                       f"</div>"

    return formatted_output

# Streamlit app setup
def top_5_items_by_month():
    st.subheader("Top 5 Items Purchased by Month and State")

    # Generate HTML output for State A and State B
    html_output = "<!DOCTYPE html><html lang='en'><head>" \
                  "<meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'>" \
                  "<title>Top Items Purchased by Month and State</title>" \
                  "<style>" \
                  "body { font-family: Arial, sans-serif; background-color: #f0f0f0; padding: 20px; }" \
                  ".state-section { margin-bottom: 30px; }" \
                  ".state-header { background-color: #343a40; color: white; padding: 10px; margin-bottom: 10px; }" \
                  ".item-list { background-color: #454d55; border: 1px solid #343a40; border-radius: 5px; padding: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }" \
                  ".item-list table { width: 100%; border-collapse: collapse; margin-top: 10px; }" \
                  ".item-list th, .item-list td { border: 1px solid #343a40; padding: 8px; text-align: left; color: white; }" \
                  ".item-list th { background-color: #343a40; }" \
                  "</style>" \
                  "</head><body>"

    html_output += format_top_items(top_items_a, 'a')
    html_output += format_top_items(top_items_b, 'b')

    html_output += "</body></html>"

    # Display the HTML output in Streamlit
    st.markdown(html_output, unsafe_allow_html=True)


# Get the unique months across both states
unique_months = pd.concat([top_items_a['Month'], top_items_b['Month']]).sort_values().unique()

# Ensure all months are included in the pivot tables
top_items_a_pivot = top_items_a.pivot_table(index='Month', columns='Item Purchased', values='Count', fill_value=0)
top_items_a_pivot = top_items_a_pivot.reindex(unique_months, fill_value=0)

top_items_b_pivot = top_items_b.pivot_table(index='Month', columns='Item Purchased', values='Count', fill_value=0)
top_items_b_pivot = top_items_b_pivot.reindex(unique_months, fill_value=0)


# Function to show heatmap of top purchase items by month
def heatmap_top_purchase_items():
    st.subheader("Heatmap of Top Purchased Items by Month and State")

    # Plot heatmap for State A using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(top_items_a_pivot, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title('Heatmap of Top Purchased Items by Month for State A')
    ax.set_xlabel('Item Purchased')
    ax.set_ylabel('Month')
    st.pyplot(fig)

    # Plot heatmap for State B using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(top_items_b_pivot, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title('Heatmap of Top Purchased Items by Month for State B')
    ax.set_xlabel('Item Purchased')
    ax.set_ylabel('Month')
    st.pyplot(fig)


# Function to provide recommendations for each state

# Function to find similar products (assuming similarity by category)
def find_similar_products(item):
    # Dummy implementation: Recommend products of the same category
    similar_items = df[df['Item Purchased'] == item]['Item Purchased'].unique()
    return similar_items

# Function to generate recommendations for each state
def generate_recommendations(state_data):
    recommendations = {}
    for month in state_data['Month'].unique():
        recommendations[month] = []
        for item in state_data[state_data['Month'] == month]['Item Purchased']:
            similar_items = find_similar_products(item)
            recommendations[month].append((item, similar_items))
    return recommendations

# Streamlit app function
def state_recommendations():
    st.subheader('State Recommendations for Top Purchased Items')

    # Separate data by state
    state_a_data = df[df['state'] == 'a']
    state_b_data = df[df['state'] == 'b']

    # Get top items for each state
    top_items_a = get_top_items_by_month(state_a_data)
    top_items_b = get_top_items_by_month(state_b_data)

    # Generate recommendations for each state
    recommendations_a = generate_recommendations(top_items_a)
    recommendations_b = generate_recommendations(top_items_b)

    # Function to format recommendations in HTML table format with dark color scheme
    def format_recommendations_html(recommendations, state_name):
        formatted_output = f"<div class='state-section'>" \
                           f"<div class='state-header'><strong>Recommendations for State {state_name.upper()}</strong></div>"

        for month, items_list in recommendations.items():
            formatted_output += f"<div class='recommendation-block'>" \
                                f"<div class='month'>Month: {month}</div>" \
                                f"<table class='recommendation-table'>" \
                                f"<tr><th style='color: #5cb85c;'>Top Purchased Item</th><th style='color: #5cb85c;'>Recommended Items</th></tr>"

            for item, similar_items in items_list:
                recommended_items = ", ".join(similar_items)
                formatted_output += f"<tr><td>{item}</td><td>{recommended_items}</td></tr>"

            formatted_output += "</table></div>"

        formatted_output += "</div>"

        return formatted_output

    # Display recommendations for State A and State B
    # Create two columns
    col1, col2 = st.columns(2)

    # Display recommendations for State A in the left column
    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(format_recommendations_html(recommendations_a, 'a'), unsafe_allow_html=True)

    # Display recommendations for State B in the right column
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(format_recommendations_html(recommendations_b, 'b'), unsafe_allow_html=True)


# FashionGenie chat bot
def fashiongenie():
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <h3>For fashion and style related queries, explore our chatbot below:</h3>
            <a href="https://mediafiles.botpress.cloud/527502e3-c141-4317-baf5-89b97c78d6c7/webchat/bot.html" 
               target="_blank" 
               style="text-decoration: none; 
                      color: white; 
                      background-color: #4CAF50; 
                      padding: 10px 20px; 
                      border-radius: 5px;">
                Open FashionGenie Chatbot
            </a>
        </div>
        """, 
        unsafe_allow_html=True
    )


# display app description
def display_app_description():
    st.title('Fashion Trend Recommender App')
    
    st.header('Challenge')
    st.write(
        "Customers often struggle to find trending fashion items specific to their region, especially during local festivals or occasions. "
        "Time-consuming searches and missed opportunities to connect with local trends can lead to decreased customer satisfaction and engagement."
    )
    
    st.header('Impact')
    st.write(
        "Time-consuming searches and missed opportunities to connect with local trends can lead to decreased customer satisfaction and engagement."
    )
    
    st.header('Need')
    st.write(
        "A solution to provide real-time, region-specific fashion recommendations to enhance the shopping experience on Myntra."
    )
    
    st.header('Objective')
    st.write(
        "Analyze real-time data to recommend trending styles based on the user’s region and provide fashion recommendations for local festivals and occasions."
    )
    
    st.header('Features')
    st.write(
        "1. **Region Trend Section:** Showcases trending products specific to a region.\n"
        "2. **AI Chatbot:** Answers fashion-related questions in a particular region and provides the product links."
    )
    
    st.header('Solution')
    st.write(
        "1. **Real-Time Data Analysis:** Utilize real-time purchase data to identify trending fashion items in specific regions.\n"
        "2. **Region-Specific Recommendations:** Create a dedicated section for 'Region Trends' where users can quickly find popular items in their area.\n"
        "3. **Occasion-Based Suggestions:** Leverage local festivals and events to tailor fashion recommendations that resonate with the cultural context of the region.\n"
        "4. **Interactive Chatbot:** A chatbot to answer fashion-related queries and suggest trends in regions unknown to the user, providing direct product links."
    )
    
    st.header('Implementation')
    st.write(
        "1. **Data Acquisition:** Used random data from Kaggle, modified to create a representative dataset.\n"
        "2. **Data Preprocessing:** Cleaned and structured the data to align with real-time analysis requirements.\n"
        "3. **Trend Analysis:** Implemented algorithms to analyze purchase patterns and identify trending items.\n"
        "4. **Recommendation Engine:** Developed a recommendation system to suggest region-specific fashion items.\n"
        "5. **Chatbot Development:** Integrated a chatbot to handle fashion queries and provide product recommendations with direct link.\n"
        "6. **User Interface:** Designed a user-friendly interface for easy access to region-specific trends."
    )
    
    st.header('Technology Stack')
    st.write(
        "1. **Chatbot:** Botpress, Myntra Products Dataset Kaggle\n"
        "2. **Backend:** Python and relevant libraries for analysis (Numpy, Pandas, Seaborn, Matplotlib, etc)\n"
        "3. **Dataset:** Fashion Retail Sales dataset Kaggle\n"
        "4. **Deployment:** GitHub"
    )
    
    st.header('Benefits to Myntra')
    st.write(
        "1. **Enhanced User Experience:** Quick access to region-specific trends saves time and improves customer satisfaction.\n"
        "2. **Increased Engagement:** Personalized recommendations and helpful chatbot encourage more frequent visits.\n"
        "3. **Boosted Sales:** Targeted suggestions during local festivals and occasions can drive higher sales.\n"
        "4. **Brand Loyalty:** Providing a tailored shopping experience fosters customer loyalty and repeat business.\n"
        "5. **Market Insights:** Real-time data analysis offers valuable insights into regional preferences and emerging trends, aiding strategic decision-making."
    )

    
# Main
# Main function to run the Streamlit app
def main():
    st.title("Fashion Data Analysis and Recommendation System")

    # Define navigation
    nav_selection = st.sidebar.radio("Navigate", 
                                     ["Display app description", "Items purchased by each state", "Monthly purchases", "Payment method by each state",
                                      "Customer segment analysis of states", "Satisfaction score",
                                      "Top 10 customers by purchase amount", "Total profit comparison for 2022 and 2023 by states",
                                      "Top 5 items purchase by months in each state",
                                      "Heatmap of top purchase items by monthly", "Recommendation for each state","FashionGenie"])

    # Display selected content based on navigation
    if nav_selection == "Display app description":
        display_app_description()
    elif nav_selection == "Items purchased by each state":
        show_items_purchased_by_state()
    elif nav_selection == "Monthly purchases":
        show_monthly_purchases()
    elif nav_selection == "Payment method by each state":
        show_payment_method_by_state()
    elif nav_selection == "Customer segment analysis of states":
        customer_segment_analysis()
    elif nav_selection == "Satisfaction score":
        display_satisfaction_scores()
    elif nav_selection == "Top 10 customers by purchase amount":
        top_10_customers()
    elif nav_selection == "Total profit comparison for 2022 and 2023 by states":
        total_profit_comparison()
    elif nav_selection == "Top 5 items purchase by months in each state":
        top_5_items_by_month()
    elif nav_selection == "Heatmap of top purchase items by monthly":
        heatmap_top_purchase_items()
    elif nav_selection == "Recommendation for each state":
        state_recommendations()
    elif nav_selection == "FashionGenie":
        fashiongenie()
    


if __name__ == "__main__":
    main()
