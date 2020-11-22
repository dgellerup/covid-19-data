#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:08:15 2020

@author: dgelleru
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import covid_analysis as cov
import covid_wi


# Workflow to create most-affected states plot
state_data = cov.load_state_data()
ma = cov.states_highest_per_capita_cases(state_data, 5)
manow = cov.get_data_since_date(ma, '2020-03-01')
cov.plot_per_capita_cases(manow)

# Workflow to create deadliest counties plot
"""
county_data = cov.load_county_data()
pcd = cov.counties_highest_per_capita_deaths(county_data, 10)
pcdnow = cov.get_data_since_date(pcd, '2020-03-15')
cov.plot_per_capita_deaths(pcdnow)
"""

# Create Cases by County gif for Wisconsin
cov.make_nice_wi_gif()

# Create New Cases plot for the Wisconsin primary election
cov.wisconsin_new_cases()

covid_wi.main()

# Create New Cases plot for Brown County, Wisconsin
#cov.brown_county_new_cases()

# Create New Cases plot for Kentucky with protest dates labeled
#cov.kentucky_protests(ma_days=3)

with open('docs/_layouts/default.html', 'r') as layout:
    lines = layout.readlines()

for i in range(len(lines)):
    if '<h2 class="project-name">' in lines[i]:
        now = datetime.datetime.now()
        newstring = '      <h2 class="project-name">Updated '
        newstring += f'{now.strftime("%A")}, {now.strftime("%B")} {now.day}</h2>\n'
        lines[i] = newstring
        
with open('docs/_layouts/default.html', 'w') as layout:
    layout.writelines(lines)



def monthly_death_comparison():
    # Data from https://www.cdc.gov/nchs/nvss/vsrr/provisional-tables.htm
    deaths_2018 = [5265, 4424, 4694, 4468, 4383, 4057, 4330, 4311, 4153, 4574, 4408, 4629]
    deaths_2019 = [4838, 4213, 4690, None, None, None, None, None, None, None, None, None]
    
    deaths = []
    for i in range(len(deaths_2018)):
        deaths.append(deaths_2018[i])
        deaths.append(deaths_2019[i])
    
    
    month = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12]
    
    years = ['2018', '2019'] * 12
    
    monthly_deaths = pd.DataFrame({'Month': month, 'Deaths': deaths, 'Year': years})
    
    sns.lineplot('Month', 'Deaths', hue='Year', data=monthly_deaths)
    plt.xticks(np.arange(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    
    
    
    