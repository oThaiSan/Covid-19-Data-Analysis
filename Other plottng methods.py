# # Plotting daily deaths in Nevada
# plt.figure(figsize=(12, 6))
# plt.plot(nevada_deaths['Date'], nevada_deaths['Daily_Deaths'], marker='o')
# plt.title('Daily COVID-19 Deaths in Nevada')
# plt.xlabel('Date')
# plt.ylabel('Daily Deaths')
# plt.grid(True)
# plt.show()





# # Bar Chart

# # Aggregate data for a specific month across all states, e.g., April 2020
# april_deaths = state_monthly_deaths[state_monthly_deaths['Month'] == '2020-04']

# plt.figure(figsize=(14, 7))
# sns.barplot(x='Province_State', y='Total_Deaths', data=april_deaths)
# plt.xticks(rotation=90)  # Rotate state names for better readability
# plt.title('COVID-19 Deaths by State in April 2020')
# plt.xlabel('State')
# plt.ylabel('Total Deaths')
# plt.show()



# # Pivot data for heat map

# heat_data = state_monthly_deaths.pivot(index="Province_State", columns="Month", values="Total_Deaths")

# plt.figure(figsize=(15, 10))
# sns.heatmap(heat_data, annot=False, cmap='viridis')  # annot set to False for cleaner visualization
# plt.title('Heat Map of COVID-19 Deaths by State and Month')
# plt.xlabel('Month')
# plt.ylabel('State')
# plt.show()


# # Box plot for all states over the entire period

# plt.figure(figsize=(14, 7))
# sns.boxplot(x='Province_State', y='Total_Deaths', data=state_monthly_deaths)
# plt.xticks(rotation=90)
# plt.title('Box Plot of Monthly COVID-19 Deaths by State')
# plt.xlabel('State')
# plt.ylabel('Total Deaths')
# plt.show()

