#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:46:19 2020

@author: dgelleru
"""

import json
import os
from typing import Iterable, List

import colorcet as cc
import geopandas as gpd
import imageio
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from plotly.figure_factory import _county_choropleth as pleth
from pylab import fill
import requests
import seaborn as sns
import us
from zipfile import ZipFile

import _covid_utils as cu

"""
st_to_state_name_dict = pleth.st_to_state_name_dict
state_to_st_dict = pleth.state_to_st_dict

us_bounds = pleth._create_us_counties_df(st_to_state_name_dict, state_to_st_dict)
us_county_bounds = us_bounds[0]
us_state_bounds = us_bounds[1]
"""

population_data = pd.read_csv('resources/population.csv')
population_data = population_data[['STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'POPESTIMATE2019']]


def get_census_api_key():
    keyfile_dir = json.load(open('api_keypath.json')).get('api_keypath')
    path_to_keyfile = os.path.join(keyfile_dir, 'keys.json')
        
    return json.load(open(path_to_keyfile)).get('us_census')


def load_national_data() -> pd.DataFrame:
    
    national_data = pd.read_csv('us.csv')
        
    new_cases = []
    new_deaths = []
    for i in range(len(national_data)):
        try:
            if national_data['cases'].iloc[i] >= national_data['cases'].iloc[i-1]:
                cdiff = national_data['cases'].iloc[i] - national_data['cases'].iloc[i-1]
            else:
                cdiff = 0
        except:
            cdiff = 0
        try:
            if national_data['deaths'].iloc[i] >= national_data['deaths'].iloc[i-1]:
                ddiff = national_data['deaths'].iloc[i] - national_data['deaths'].iloc[i-1]
            else:
                ddiff = 0
        except:
            ddiff = 0
        
        new_cases.append(cdiff)
        new_deaths.append(ddiff)
    
    national_data['new_cases'] = new_cases
    national_data['new_deaths'] = new_deaths
    
    national_population = population_data[population_data['COUNTY'] == 0]
    national_population = national_population['POPESTIMATE2019'].sum()
    
    national_data['population'] = national_population
    national_data['cases_per_100k'] = national_data['cases'] / (national_data['population']/100000)
    national_data['deaths_per_100k'] = national_data['deaths'] / (national_data['population']/100000)
    
    national_data['case_lethality'] = national_data['deaths'] / national_data['cases']
    
    return national_data

    
def load_state_data() -> pd.DataFrame:
    state_data = pd.read_csv('us-states.csv')
    
    state_data.sort_values(['state', 'date'], inplace=True)
    
    new_cases = []
    new_deaths = []
    for i in range(len(state_data)):
        try:
            if state_data['cases'].iloc[i] >= state_data['cases'].iloc[i-1]:
                cdiff = state_data['cases'].iloc[i] - state_data['cases'].iloc[i-1]
            else:
                cdiff = 0
        except:
            cdiff = 0
        try:
            if state_data['deaths'].iloc[i] >= state_data['deaths'].iloc[i-1]:
                ddiff = state_data['deaths'].iloc[i] - state_data['deaths'].iloc[i-1]
            else:
                ddiff = 0
        except:
            ddiff = 0
            
        new_cases.append(cdiff)
        new_deaths.append(ddiff)
    
    state_data['new_cases'] = new_cases
    state_data['new_deaths'] = new_deaths
    
    state_data.sort_index(inplace=True)
    
    state_population = population_data[population_data['COUNTY'] == 0]
    state_population.drop(['STATE', 'COUNTY', 'CTYNAME'], axis=1, inplace=True)
    state_population.rename(columns={'STNAME': 'state', 'POPESTIMATE2019': 'population'}, inplace=True)
    
    final_state_data = state_data.join(state_population.set_index('state'), on='state')
    
    final_state_data['cases_per_100k'] = final_state_data['cases'] / (final_state_data['population']/100000)
    final_state_data['deaths_per_100k'] = final_state_data['deaths'] / (final_state_data['population']/100000)
        
    return final_state_data


def load_county_data() -> pd.DataFrame:
    county_data = pd.read_csv('us-counties.csv')
    county_data['county, state'] = county_data[['county', 'state']].apply(lambda x: f"{x['county']}, {x['state']}", axis=1)
    
    county_data.sort_values(['county, state', 'date'], inplace=True)
    
    new_cases = []
    new_deaths = []
    for i in range(len(county_data)):
        try:
            if county_data['cases'].iloc[i] >= county_data['cases'].iloc[i-1]:
                cdiff = county_data['cases'].iloc[i] - county_data['cases'].iloc[i-1]
            else:
                cdiff = 0
        except:
            cdiff = 0
        try:
            if county_data['deaths'].iloc[i] >= county_data['deaths'].iloc[i-1]:
                ddiff = county_data['deaths'].iloc[i] - county_data['deaths'].iloc[i-1]
            else:
                ddiff = 0
        except:
            ddiff = 0
        new_cases.append(cdiff)
        new_deaths.append(ddiff)
    
    county_data['new_cases'] = new_cases
    county_data['new_deaths'] = new_deaths
    
    county_data.sort_index(inplace=True)
    
    county_data.dropna(inplace=True)
    
    county_data['fips'] = county_data['fips'].astype(int).astype(str).apply(lambda x: f'0{x}' if len(x) == 4 else x)
    
    county_population = population_data[population_data['COUNTY'] != 0]
    
    state_fips = population_data['STATE'].astype(str).apply(lambda x: f'0{x}' if len(x) < 2 else x)
    county_fips = population_data['COUNTY'].astype(str).apply(lambda x: f'00{x}' if len(x) == 1 else (f'0{x}' if len(x) == 2 else x))
    
    county_population.drop(['STATE', 'COUNTY', 'STNAME', 'CTYNAME'], axis=1, inplace=True)
    county_population.rename(columns={'POPESTIMATE2019': 'population'}, inplace=True)
    
    county_population['fips'] = state_fips + county_fips
    
    final_county_data = county_data.join(county_population.set_index('fips'), on='fips')
    
    final_county_data['cases_per_10k'] = final_county_data['cases'] / (final_county_data['population']/10000)
    final_county_data['deaths_per_10k'] = final_county_data['deaths'] / (final_county_data['population']/10000)
    
    final_county_data['fips'] = final_county_data['fips'].astype(int)
    
    return final_county_data


def select_states(df: pd.DataFrame, states: str or List[str]) -> pd.DataFrame:
    """
    Accepts a Pandas DataFrame and a string or list of strings containing state names.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing state data.
    states : str or List[str]
        Names of states to select.

    Returns
    -------
    states_df : TYPE
        Pandas DataFrame only containing data for the selected states.

    """
    
    if type(states) == str:
        states = [states]
        
    states_df = df[df['state'].isin(states)]
    
    return states_df


def states_with_most_cases(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    most_recent_df.sort_values('cases', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_states_full_data = df[df['state'].isin(most_affected['state'])]
    
    return most_affected_states_full_data


def counties_with_most_cases(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    most_recent_df.sort_values('cases', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_counties_full_data = df[df['county, state'].isin(most_affected['county, state'])]
    
    return most_affected_counties_full_data


def states_with_most_deaths(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    most_recent_df.sort_values('deaths', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_states_full_data = df[df['state'].isin(most_affected['state'])]
    
    return most_affected_states_full_data


def counties_with_most_deaths(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    most_recent_df.sort_values('deaths', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_counties_full_data = df[df['county, state'].isin(most_affected['county, state'])]
    
    return most_affected_counties_full_data


def states_highest_per_capita_cases(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    if 'county' in df.columns:
        most_recent_df.sort_values('cases_per_10k', ascending=False, inplace=True)
    else:
        most_recent_df.sort_values('cases_per_100k', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_states_full_data = df[df['state'].isin(most_affected['state'])]
    
    return most_affected_states_full_data


def states_highest_per_capita_deaths(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    most_recent_df.sort_values('deaths_per_100k', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_states_full_data = df[df['state'].isin(most_affected['state'])]
    
    return most_affected_states_full_data


def counties_highest_per_capita_cases(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    most_recent_df.sort_values('cases_per_10k', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_counties_full_data = df[df['county, state'].isin(most_affected['county, state'])]
    
    return most_affected_counties_full_data


def counties_highest_per_capita_deaths(df: pd.DataFrame, number: int=5) -> pd.DataFrame:
    most_recent_date = df['date'].max()
    
    most_recent_df = pd.DataFrame(df[df['date'] == most_recent_date])
    most_recent_df.sort_values('deaths_per_10k', ascending=False, inplace=True)
    most_recent_df.reset_index(drop=True, inplace=True)
    
    most_affected = pd.DataFrame(most_recent_df.iloc[:number])
    most_affected_counties_full_data = df[df['county, state'].isin(most_affected['county, state'])]
    
    return most_affected_counties_full_data


def get_date_of_first_case(df: pd.DataFrame, state: str="Washington") -> str:
    state_df = pd.DataFrame(df[df['state'] == state])
    state_df = state_df[state_df['cases'] != 0]
    earliest_case = state_df.iloc[0]['date']
    return earliest_case


def get_date_of_first_death(df: pd.DataFrame, state: str="Washington") -> str:
    state_df = pd.DataFrame(df[df['state'] == state])
    state_df = state_df[state_df['deaths'] != 0]
    earliest_case = state_df.iloc[0]['date']
    return earliest_case


def get_data_since_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
    data_after_date = df[df['date'] >= date]
    return data_after_date


def three_day_moving_average(df: pd.DataFrame, metric: str="new_cases") -> pd.DataFrame:
    """
    Creates new column 'moving_ave' and populates it with the five day moving
    average of new cases.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing state data.

    Returns
    -------
    List[float]
        Pandas DataFrame with 'moving_ave' column/data added.

    """
    
    columns = list(df.columns)
    
    if 'state' in columns:
        columns.append('moving_ave')
        
        #master_df = pd.DataFrame(columns=columns)
        master_list = []
        for state in list(set(df['state'])):
            state_df = df[df['state'] == state]
            
            iterable = state_df[metric]
            
            casted = list(iterable)
            
            moving_ave = []
            
            for i in range(len(casted)):
                if i > (len(casted) - 2) or i < 1:
                    moving_ave.append(None)
                else:
                    before_after = casted[i-1:i+2]
                    moving_ave.append(pd.Series(before_after).mean())
                    
            state_df['moving_ave'] = moving_ave
        
            #master_df = master_df.append(state_df)
            master_list.append(state_df)
        
        master_df = pd.concat(master_list)
        
        master_df.sort_index()
        
        return master_df
    
    else:
        iterable = df[metric]
        
        casted = list(iterable)
        
        moving_ave = []
        for i in range(len(casted)):
            if i > (len(casted) - 2) or i < 1:
                moving_ave.append(None)
            else:
                before_after = casted[i-1:i+2]
                moving_ave.append(pd.Series(before_after).mean())
            
        df['moving_ave'] = moving_ave
        
        return df
    
    
def five_day_moving_average(df: pd.DataFrame, metric: str="new_cases") -> pd.DataFrame:
    """
    Creates new column 'moving_ave' and populates it with the five day moving
    average of new cases.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing state data.

    Returns
    -------
    List[float]
        Pandas DataFrame with 'moving_ave' column/data added.

    """
    
    columns = list(df.columns)
    
    if 'county' in columns:
        columns.append('moving_ave')
        
        master_list = []
        for county in list(set(df['county'])):
            county_df = df[df['county'] == county]
            
            iterable = county_df[metric]
            
            casted = list(iterable)
            
            moving_ave = []
            
            for i in range(len(casted)):
                if i > (len(casted) - 3) or i < 2:
                    moving_ave.append(None)
                else:
                    before_after = casted[i-2:i+3]
                    moving_ave.append(pd.Series(before_after).mean())
            
            county_df['moving_ave'] = moving_ave
            
            master_list.append(county_df)
            
        master_df = pd.concat(master_list)
        
        master_df.sort_index(inplace=True)
        
        return master_df
            
    elif 'state' in columns:
        columns.append('moving_ave')
        
        #master_df = pd.DataFrame(columns=columns)
        master_list = []
        for state in list(set(df['state'])):
            state_df = df[df['state'] == state]
            
            iterable = state_df[metric]
            
            casted = list(iterable)
            
            moving_ave = []
            
            for i in range(len(casted)):
                if i > (len(casted) - 3) or i < 2:
                    moving_ave.append(None)
                else:
                    before_after = casted[i-2:i+3]
                    moving_ave.append(pd.Series(before_after).mean())
                    
            state_df['moving_ave'] = moving_ave
        
            #master_df = master_df.append(state_df)
            master_list.append(state_df)
        
        master_df = pd.concat(master_list)
        
        master_df.sort_index(inplace=True)
        
        return master_df
    
    else:
        iterable = df[metric]
        
        casted = list(iterable)
        
        moving_ave = []
        for i in range(len(casted)):
            if i > (len(casted) - 3) or i < 2:
                moving_ave.append(None)
            else:
                before_after = casted[i-2:i+3]
                moving_ave.append(pd.Series(before_after).mean())
            
        df['moving_ave'] = moving_ave
        
        return df


def seven_day_moving_average(df: pd.DataFrame, metric: str="new_cases") -> pd.DataFrame:
    """
    Creates new column 'moving_ave' and populates it with the seven day moving
    average of new cases.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing state data.

    Returns
    -------
    List[float]
        Pandas DataFrame with 'moving_ave' column/data added.

    """
    
    columns = list(df.columns)
    
    if 'state' in columns:
        columns.append('moving_ave')
        
        master_df = pd.DataFrame(columns=columns)
        
        for state in list(set(df['state'])):
            state_df = df[df['state'] == state]
            
            iterable = state_df[metric]
            
            casted = list(iterable)
            
            moving_ave = []
            
            for i in range(len(casted)):
                if i > (len(casted) - 4) or i < 3:
                    moving_ave.append(None)
                else:
                    before_after = casted[i-3:i+4]
                    moving_ave.append(pd.Series(before_after).mean())
                    
            state_df['moving_ave'] = moving_ave
        
            master_df = master_df.append(state_df)
        
        master_df.sort_index()
        
        return master_df
    
    else:
        iterable = df[metric]
        
        casted = list(iterable)
        
        moving_ave = []
        for i in range(len(casted)):
            if i > (len(casted) - 4) or i < 3:
                moving_ave.append(None)
            else:
                before_after = casted[i-3:i+4]
                moving_ave.append(pd.Series(before_after).mean())
            
        df['moving_ave'] = moving_ave
        
        return df
    
    
def nine_day_moving_average(df: pd.DataFrame, metric: str="new_cases") -> pd.DataFrame:
    """
    Creates new column 'moving_ave' and populates it with the nine day moving
    average of new cases.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing state data.

    Returns
    -------
    List[float]
        Pandas DataFrame with 'moving_ave' column/data added.

    """
    
    columns = list(df.columns)
    
    if 'state' in columns:
        columns.append('moving_ave')
        
        master_df = pd.DataFrame(columns=columns)
        
        for state in list(set(df['state'])):
            state_df = df[df['state'] == state]
            
            iterable = state_df[metric]
            
            casted = list(iterable)
            
            moving_ave = []
            
            for i in range(len(casted)):
                if i > (len(casted) - 5) or i < 4:
                    moving_ave.append(None)
                else:
                    before_after = casted[i-4:i+5]
                    moving_ave.append(pd.Series(before_after).mean())
                    
            state_df['moving_ave'] = moving_ave
        
            master_df = master_df.append(state_df)
        
        master_df.sort_index()
        
        return master_df
    
    else:
        iterable = df[metric]
        
        casted = list(iterable)
        
        moving_ave = []
        for i in range(len(casted)):
            if i > (len(casted) - 5) or i < 4:
                moving_ave.append(None)
            else:
                before_after = casted[i-4:i+5]
                moving_ave.append(pd.Series(before_after).mean())
            
        df['moving_ave'] = moving_ave
        
        return df
    

def fit_curve(iterable: Iterable, deg: int) -> List[int]:
    "This more or less does not work but I want to leave it here for future ref."
    x = range(0, len(iterable))
    x_future = range(0, len(iterable)+14)
    y = iterable
    
    z = np.polyfit(x, y, deg)
    
    p = np.poly1d(z)
    
    fit = [int(p(f)) for f in x_future]
    
    return fit


def plot_cases(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        cu._plot_county_cases(df)
    elif "state" not in df.columns:
        cu._plot_national_cases(df)
    else:
        cu._plot_state_cases(df)
        
        
def plot_deaths(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        cu._plot_county_deaths(df)
    elif 'state' not in df.columns:
        cu._plot_national_deaths(df)
    else:
        cu._plot_state_deaths(df)
                

def plot_per_capita_cases(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        cu._plot_cases_per_capita_county(df)
    elif "state" not in df.columns:
        cu._plot_cases_per_capita_national(df)
    else:
        cu._plot_cases_per_capita_state(df)
    
    
def plot_per_capita_deaths(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        cu._plot_deaths_per_capita_county(df)
    elif "state" not in df.columns:
        cu._plot_deaths_per_capita_national(df)
    else:
        cu._plot_deaths_per_capita_state(df)
        
        
def plot_new_cases(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        cu._plot_new_cases_county(df)
    elif "state" not in df.columns:
        cu._plot_new_cases_national(df)
    else:
        cu._plot_new_cases_state(df)
        
        
def plot_new_deaths(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        cu._plot_new_deaths_county(df)
    elif "state" not in df.columns:
        cu._plot_new_deaths_national(df)
    else:
        cu._plot_new_deaths_state(df)
        

def plot_state_cases_vs_deaths(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.lineplot('date', 'cases', data=df, color="b", legend=None)
    plt.ylabel('Cases')
    plt.xticks(rotation=90)
    ax2 = plt.twinx()
    sns.lineplot('date', 'deaths', data=df, color="r", ax=ax2, legend=None)
    
    
    blue_line = mlines.Line2D([], [], color='blue', marker=None, label='Cases')
    red_line = mlines.Line2D([], [], color='red', marker=None, label="Deaths")
    plt.legend(handles=[blue_line, red_line])
    #plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=1)
    #plt.legend(loc='upper left')
    
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    plt.tight_layout()


def plot_moving_average(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        cu._plot_county_moving_average(df)
    elif "state" not in df.columns:
        cu._plot_national_moving_average(df)
    else:
        cu._plot_state_moving_average(df)
    

def wisconsin_new_cases(df: pd.DataFrame=None, ma_days: int=5) -> None:
    try:
        dummy = len(df)
    except:
        state_data = load_state_data()
        wi = select_states(state_data, 'Wisconsin')
        df = get_data_since_date(wi, '2020-03-08')
        
    if 'moving_ave' not in df.columns:
        if ma_days not in [3, 5, 7, 9]:
            df = five_day_moving_average(df)
        else:
            if ma_days == 3:
                df = three_day_moving_average(df)
            elif ma_days == 5:
                df = five_day_moving_average(df)
            elif ma_days == 7:
                df = seven_day_moving_average(df)
            elif ma_days == 9:
                df = nine_day_moving_average(df)
                
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    sns.lineplot('date', 'moving_ave', hue='state', data=df, legend=None)
    plt.ylabel(f'New Cases - {ma_days} day moving average', fontsize=16)
    x_ticks = [tick for tick in plt.xticks()[0] if tick % 4 == 0]
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    plt.xticks(size=10, rotation=90)
    
    plt.axvline('2020-04-07', color='red', linestyle='--')
    plt.axvline('2020-05-13', color='red', linestyle='--')
    plt.axvline('2020-08-01', color='red', linestyle='--')
    plt.axvline('2020-09-02', color='red', linestyle='--')
    plt.text('2020-03-30', 300, 'Wisconsin Primary')
    plt.text('2020-04-28', 50, 'Stay At Home Order Overturned')
    plt.text('2020-07-18', 400, 'Mask Mandate Enacted')
    plt.text('2020-08-19', 200, 'UW System Instruction Begins')
    plt.plot(['2020-04-04', '2020-04-07'], [300, 250], 'black', linewidth=1)
    plt.plot(['2020-05-10', '2020-05-13'], [50, 25], 'black', linewidth=1)
    plt.plot(['2020-07-27', '2020-08-01'], [400, 325], 'black', linewidth=1)
    plt.plot(['2020-08-28', '2020-09-02'], [200, 150], 'black', linewidth=1)
    
    max_date = df['date'].max()
    split_date = max_date.split("-")
    
    incubation_end = f"{'-'.join(split_date[:2])}-{str(int(split_date[-1])-int(ma_days/2))}" if df['date'].max() <= '2020-04-22' else '2020-04-21'
    fill(['2020-04-08', incubation_end, incubation_end, '2020-04-08'], 
         [0,0,df['moving_ave'].max(),df['moving_ave'].max()],
         'r', alpha=0.2, edgecolor='r')
        
    #plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=1)
    plt.xlabel('Date', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/wisconsin_new_cases.svg', dpi=300)
    

def brown_county_new_cases(df: pd.DataFrame=None, ma_days: int=5) -> None:
    # This is a dumb way of checking if a pd.DataFrame was provided.
    try:
        dummy = len(df)
    except:
        county_data = load_county_data()
        wi = select_states(county_data, 'Wisconsin')
        winow = get_data_since_date(wi, '2020-03-08')
        df = winow[winow['county'].isin(['Brown'])]
    
    if 'moving_ave' not in df.columns:
        if ma_days not in [3, 5, 7, 9]:
            df = five_day_moving_average(df)
        else:
            if ma_days == 3:
                df = three_day_moving_average(df)
            elif ma_days == 5:
                df = five_day_moving_average(df)
            elif ma_days == 7:
                df = seven_day_moving_average(df)
            elif ma_days == 9:
                df = nine_day_moving_average(df)
                
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 6)
    sns.lineplot('date', 'moving_ave', hue='county', data=df, legend=False)
    plt.ylabel(f'New Cases - {ma_days} day moving average', fontsize=16)
    x_ticks = [tick for tick in plt.xticks()[0] if tick % 4 == 0]
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_ticks))
    plt.xticks(size=10, rotation=90)
    plt.axvline('2020-04-07', color='red', linestyle='--')
    plt.text('2020-04-01', 15, 'Wisconsin Primary')
    plt.plot(['2020-04-04', '2020-04-07'], [15, 10], 'black', linewidth=1)
    max_date = df['date'].max()
    split_date = max_date.split("-")
    incubation_end = f"{'-'.join(split_date[:2])}-{str(int(split_date[-1])-int(ma_days/2))}" if df['date'].max() <= '2020-04-22' else '2020-04-21'
    fill(['2020-04-08',incubation_end,incubation_end,'2020-04-08'], [0,0,df['moving_ave'].max(),df['moving_ave'].max()], 'r', alpha=0.2, edgecolor='r')
    #plt.text('2020-04-09', 15, 'Incubation Period')
    
    plt.title('')
    plt.xlabel('Date', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/brown_county_new_cases.svg', dpi=300)
    

def kentucky_protests(df: pd.DataFrame=None, ma_days: int=5) -> None:
    # This is a dumb way of checking if a pd.DataFrame was provided.
    try:
        dummy = len(df)
    except:
        state_data = load_state_data()
        ky = select_states(state_data, 'Kentucky')
        df = get_data_since_date(ky, '2020-03-06')
    
    if 'moving_ave' not in df.columns:
        if ma_days not in [3, 5, 7, 9]:
            df = five_day_moving_average(df)
        else:
            if ma_days == 3:
                df = three_day_moving_average(df)
            elif ma_days == 5:
                df = five_day_moving_average(df)
            elif ma_days == 7:
                df = seven_day_moving_average(df)
            elif ma_days == 9:
                df = nine_day_moving_average(df)
                
    plt.figure(figsize=(12, 6))
    sns.lineplot('date', 'moving_ave', hue='state', data=df, legend=None)
    plt.ylabel(f'New Cases - {ma_days} day moving average')
    plt.xticks(rotation=90)
    plt.axvline('2020-04-17', color='red', linestyle='--')
    plt.axvline('2020-04-15', color='red', linestyle='--')
    plt.text('2020-04-11', 60, 'Kentucky Protests')
    plt.plot(['2020-04-14', '2020-04-17'], [65, 80], 'black', linewidth=1)
    plt.plot(['2020-04-14', '2020-04-15'], [65, 80], 'black', linewidth=1)
    
    plt.title('Kentucky New Cases')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig('plots/consequences_ky.png', dpi=300)    
        
    
def make_state_counties_gif(state: str, date: str='default') -> None:
    
    county_data = load_county_data()
    state_df = county_data[county_data['state'] == us.states.lookup(state).name]
    
    if date == 'default':
        state_df = get_data_since_date(state_df, get_date_of_first_case(state_df, state))
    else:
        state_df = get_data_since_date(state_df, date)
        
    vmin, vmax = 0, state_df['cases'].max()
    
    shapefile_url = us.states.lookup(state).shapefile_urls().get('county')
    
    r = requests.get(shapefile_url)
    
    basepath = os.path.join(os.getcwd(), f'resources/shapefiles/{state}')
    plotpath = os.path.join(os.getcwd(), f'plots/{us.states.lookup(state).name}')
    
    if not os.path.exists(basepath):
        os.makedirs(basepath)
        
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)
    
    with open(os.path.join(basepath, 'shp.zip'), 'wb') as f:
        f.write(r.content)
    
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(os.path.join(basepath, 'shp.zip'), 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall(basepath)
    
    """map_df = gpd.read_file('zip://resources/shapefiles/c_02ap19.zip')
    
    map_df = map_df[map_df['STATE'] == us.states.lookup(state).abbr]"""
    
    shapefile = list(filter(lambda x: x.endswith('.shp'), os.listdir(basepath)))[0]
    
    shapefile_path = os.path.join(basepath, shapefile)
    
    map_df = gpd.read_file(shapefile_path)
    #map_df['COUNTY_FIP'] = map_df['COUNTY_FIP'].astype(int)
    
    images = []
    
    for plot_date in list(set(state_df['date'])):
        
        state_from_date = state_df[state_df['date'] == plot_date]
        
        state_from_date['fips'] = state_from_date['fips'].astype(str)

        merged = map_df.set_index("GEOID10").join(state_from_date.set_index('fips'))
        
        variable = 'cases'
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        
        merged.plot(column=variable,
                    cmap='viridis',
                    linewidth=0.8,
                    ax=ax,
                    edgecolor='0.8',
                    vmin=vmin,
                    vmax=vmax,
                    missing_kwds={"color": "lightgrey"})
        
        ax.axis('off')
        
        # add a title
        ax.set_title(f"Cases by County {plot_date}",
                     fontdict={'fontsize': '18', 'fontweight' : '3'})
        
        # create an annotation for the data source
        ax.annotate('Source: The New York Times, 2020',
                    xy=(0.1, 0.08),
                    xycoords='figure fraction',
                    horizontalalignment='left', 
                    verticalalignment='top',
                    fontsize=12,
                    color='#555555')
        
        # Create colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        
        # empty array for the data range
        sm._A = []
        
        # add the colorbar to the figure
        cbar = fig.colorbar(sm)
        
        #plt.axis('equal')
        
        fig.savefig(os.path.join(plotpath, f'counties_{plot_date}.png'), dpi=300)
        
        plt.close(fig)
        
        images.append(os.path.join(plotpath, f'counties_{plot_date}.png'))
        
    images = sorted(images)
    
    gif_images = [imageio.imread(x) for x in images]
    
    imageio.mimsave(f'plots/{state}.gif', gif_images, duration=0.5)
    
    
def make_nice_wi_gif(new_cases=False):
    
    state_data = load_state_data()
    state_data = state_data[state_data['state'] == 'Wisconsin']
    wi_state = get_data_since_date(state_data, '2020-03-08')
    
    county_data = load_county_data()
    state_df = county_data[county_data['state'] == 'Wisconsin']
            
    state_df['short_fips'] = state_df['fips'].apply(lambda x: int(str(int(x))[2:]))
    
    state_df = get_data_since_date(state_df, '2020-03-08')
    
    state_df['new_cases_per_10k'] = state_df['new_cases'] / (state_df['population']/10000)
    
    state_df['cases_per_10k'] = state_df['cases'] / (state_df['population']/10000)
    
    if new_cases:
        vmin, vmax = 0, state_df['new_cases_per_10k'].max()
        colormap = 'Reds'
    else:
        vmin, vmax = 0, state_df['cases_per_10k'].max()
        colormap = 'Reds'
        
    basepath = os.path.join(os.getcwd(), 'resources/shapefiles/Wisconsin')
    plotpath = os.path.join(os.getcwd(), 'plots/wisconsin')
    
    if not os.path.exists(basepath):
        os.makedirs(basepath)
        
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)
        
    shapefile = list(filter(lambda x: x.endswith('.shp'), os.listdir(basepath)))[0]
    
    shapefile_path = os.path.join(basepath, shapefile)
    
    map_df = gpd.read_file(shapefile_path)
    map_df['COUNTY_FIP'] = map_df['COUNTY_FIP'].astype(int)
        
    images = []
    
    plot_dates = sorted(list(set(state_df['date'])))
    plot_dates = [plot_dates[i] for i in range(0, len(plot_dates), 3)]
    
    for date in plot_dates:
    
        wi_state['era'] = wi_state['date'].apply(lambda x: 'old' if x <= date else 'new')
        
        winow = state_df[state_df['date'] == date]

        merged = map_df.set_index("COUNTY_FIP").join(winow.set_index('short_fips'))
        
        #merged['cases'] = merged['cases'].apply(lambda x: 0 if pd.isnull(x) else x)
        
        variable = 'cases_per_10k' if new_cases == False else 'new_cases_per_10k'
        
        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(9, 9))
        
        ax0.set_aspect('equal')
        
        merged.plot(column=variable,
                    cmap=colormap,
                    linewidth=0.8,
                    ax=ax0,
                    edgecolor='0.8',
                    vmin=vmin,
                    vmax=vmax,
                    missing_kwds={"color": "lightgrey"},
                    figsize=(6, 2))
        
        ax0.axis('off')
        
        # add a title
        if new_cases:
            ax0.set_title(f"New Cases per 10K People {date}",
                          fontdict={'fontsize': '18', 'fontweight': '3'})
        else:
            ax0.set_title(f"Cases per 10K People {date}",
                          fontdict={'fontsize': '18', 'fontweight': '3'})
        
        # create an annotation for the data source
        ax0.annotate('Source: The New York Times, 2020',
                    xy=(0.1, 0.08),
                    xycoords='figure fraction',
                    horizontalalignment='left', 
                    verticalalignment='top',
                    fontsize=12,
                    color='#555555')
        
        # Create colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        
        # empty array for the data range
        sm._A = []
        
        # add the colorbar to the figure
        cbar = fig.colorbar(sm, ax=ax0)
        
        sns.lineplot('date', 'cases', data=wi_state, ax=ax1, color='coral')
        plt.fill_between(wi_state.date.values, wi_state.cases.values, color='coral')
        plt.axvline(date)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        ax1.set_ylabel('State Cases')
        
        fig.savefig(os.path.join(plotpath, f'counties_{date}.png'), dpi=300)
        
        
        plt.close(fig)
            
        images.append(os.path.join(plotpath, f'counties_{date}.png'))
        
    images = sorted(images)
    
    gif_images = [imageio.imread(image) for image in images]
    
    
    
    imageio.mimsave('plots/wisconsin_new_cases.gif', gif_images, duration=0.5)
    
        
    
    
    
    
def make_us_counties_gif(date: str='default') -> None:
    
    county_data = load_county_data()
    
    county_data['fips'] = county_data['fips'].astype(str)
    
    county_data['fips'] = county_data['fips'].apply(fix_fips)
    
    if date == 'default':
        county_data = get_data_since_date(county_data, get_date_of_first_case(county_data, 'Washington'))
    else:
        county_data = get_data_since_date(county_data, date)
        
    vmin, vmax = 0, county_data['cases'].max()
    
    #shapefile_url = us.states.lookup(state).shapefile_urls().get('county')
    
    #r = requests.get(shapefile_url)
    
    #basepath = os.path.join(os.getcwd(), f'resources/shapefiles/{state}')
    plotpath = os.path.join(os.getcwd(), 'plots/us')
    
    """if not os.path.exists(basepath):
        os.makedirs(basepath)"""
        
    if not os.path.exists(plotpath):
        os.makedirs(plotpath)
    
    """with open(os.path.join(basepath, 'shp.zip'), 'wb') as f:
        f.write(r.content)
    
    # Create a ZipFile Object and load sample.zip in it
    with ZipFile(os.path.join(basepath, 'shp.zip'), 'r') as zipObj:
       # Extract all the contents of zip file in current directory
       zipObj.extractall(basepath)"""
    
    """map_df = gpd.read_file('zip://resources/shapefiles/c_02ap19.zip')
    
    map_df = map_df[map_df['STATE'] == us.states.lookup(state).abbr]"""
    
    #shapefile = list(filter(lambda x: x.endswith('.shp'), os.listdir(basepath)))[0]
    
    #shapefile_path = os.path.join(basepath, shapefile)
    
    #map_df = gpd.read_file(shapefile_path)
    #map_df['COUNTY_FIP'] = map_df['COUNTY_FIP'].astype(int)
    map_df = gpd.GeoDataFrame(us_county_bounds)
    map_df = map_df[map_df['GEOID'].isin(list(set(county_data['fips'])))] # get rid of territories
    map_df = map_df[map_df['GEOID'].str.startswith('15') == False] # get rid of Hawaii :(
    map_df = map_df[map_df['GEOID'].str.startswith('02') == False]
    #map_df = map_df[map_df['STATE_NAME'] == state]
    
    images = []
    
    for plot_date in list(set(state_df['date'])):
        
        us_from_date = county_data[county_data['date'] == plot_date]

        merged = map_df.set_index("GEOID").join(us_from_date.set_index('fips'))
        
        variable = 'cases'
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        merged.plot(column=variable,
                    cmap='viridis',
                    linewidth=0.8,
                    ax=ax,
                    edgecolor='0.8',
                    vmin=vmin,
                    vmax=vmax,
                    missing_kwds={"color": "lightgrey"})
        
        ax.axis('off')
        
        # add a title
        ax.set_title(f"Cases by County {plot_date}",
                     fontdict={'fontsize': '18', 'fontweight' : '3'})
        
        # create an annotation for the data source
        ax.annotate('Source: The New York Times, 2020',
                    xy=(0.1, 0.08),
                    xycoords='figure fraction',
                    horizontalalignment='left', 
                    verticalalignment='top',
                    fontsize=12,
                    color='#555555')
        
        # Create colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        
        # empty array for the data range
        sm._A = []
        
        # add the colorbar to the figure
        cbar = fig.colorbar(sm)
        
        #plt.axis('equal')
        plt.show()
        
        fig.savefig(os.path.join(plotpath, f'counties_{plot_date}.png'), dpi=300)
        
        plt.close(fig)
        
        images.append(os.path.join(plotpath, f'counties_{plot_date}.png'))
        
    images = sorted(images)
    
    gif_images = [imageio.imread(x) for x in images]
    
    imageio.mimsave(f'plots/{state}.gif', gif_images, duration=0.5)
    
    
    
    
    
    
    
    
    

    