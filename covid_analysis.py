#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:46:19 2020

@author: dgelleru
"""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_state_data() -> pd.DataFrame:
    state_data = pd.read_csv('us-states.csv')
    
    state_data.sort_values(['state', 'date'], inplace=True)
    
    new_cases = []
    new_deaths = []
    for i in range(len(state_data)):
        try:
            if state_data['cases'].iloc[i] >= state_data['cases'].iloc[i-1]:
                cdiff = state_data['cases'].iloc[i] - state_data['cases'].iloc[i-1]
                ddiff = state_data['deaths'].iloc[i] - state_data['deaths'].iloc[i-1]
            else:
                cdiff = state_data['cases'].iloc[i]
                ddiff = state_data['deaths'].iloc[i]
        except:
            cdiff = state_data['cases'].iloc[i]
            ddiff = state_data['deaths'].iloc[i]
            
        new_cases.append(cdiff)
        new_deaths.append(ddiff)
    
    state_data['new_cases'] = new_cases
    state_data['new_deaths'] = new_deaths
    
    state_data.sort_index(inplace=True)
        
    return state_data


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
                ddiff = county_data['deaths'].iloc[i] - county_data['deaths'].iloc[i-1]
            else:
                cdiff = county_data['cases'].iloc[i]
                ddiff = county_data['deaths'].iloc[i]
        except:
            cdiff = county_data['cases'].iloc[i]
            ddiff = county_data['deaths'].iloc[i]
        new_cases.append(cdiff)
        new_deaths.append(ddiff)
    
    county_data['new_cases'] = new_cases
    county_data['new_deaths'] = new_deaths
    
    county_data.sort_index(inplace=True)
    
    return county_data


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
    most_recent_df.sort_values('casess', ascending=False, inplace=True)
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


def five_day_moving_average(iterable):
    casted = list(iterable)
    
    moving_ave = []
    for i in range(len(casted)):
        if 1 > i > (len(casted) - 2):
            moving_ave.append(None)
        else:
            before_after = casted[i-2:i]
            before_after.extend(casted[i:i+3])
            moving_ave.append(pd.Series(before_after).mean())
    return moving_ave
    
    
def nine_day_moving_average(iterable):
    casted = list(iterable)
    
    moving_ave = []
    for i in range(len(casted)):
        if 3 > i > (len(casted) - 5):
            moving_ave.append(None)
        else:
            before_after = casted[i-4:i]
            before_after.extend(casted[i:i+5])
            moving_ave.append(pd.Series(before_after).mean())
    return moving_ave


def fit_curve(iterable, deg):
    "This more or less does not work but I want to leave it here for future ref."
    x = range(0, len(iterable))
    y = iterable
    
    z = np.polyfit(x, y, deg)
    
    p = np.poly1d(z)
    
    fit = [int(p(f)) for f in x]
    
    return fit


def plot_cases(df):
    if "county, state" in df.columns:
        _plot_county_cases(df)
    else:
        _plot_state_cases(df)
        
        
def plot_deaths(df):
    if "county, state" in df.columns:
        _plot_county_deaths(df)
    else:
        _plot_state_deaths(df)


def _plot_state_cases(df):
    
    num_states = len(list(set(df['state'])))
    
    if num_states > 27:
        num_col = 2
    else:
        num_col = 1
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'cases', hue='state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('Cases')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_state_deaths(df):
    
    num_states = len(list(set(df['state'])))
    
    if num_states > 27:
        num_col = 2
    else:
        num_col = 1
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'deaths', hue='state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_county_cases(df):
    
    num_counties = len(list(set(df['county'])))
    
    if num_counties > 27:
        num_col = int(num_counties/27) + 1
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'cases', hue='county, state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('Cases')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_county_deaths(df):
    
    num_counties = len(list(set(df['county'])))
    
    if num_counties > 27:
        num_col = int(num_counties/27) + 1
    else:
        num_col = 1
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'deaths', hue='county, state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    plt.tight_layout()


def plot_new_cases(df):
    if "county, state" in df.columns:
        _plot_new_cases_county(df)
    else:
        _plot_new_cases_state(df)
        
        
def plot_new_deaths(df):
    if "county, state" in df.columns:
        _plot_new_deaths_county(df)
    else:
        _plot_new_deaths_state(df)
        
    
def _plot_new_cases_state(df):
    num_states = len(list(set(df['state'])))
    
    if num_states > 27:
        num_col = 2
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'new_cases', hue='state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('New Cases')
    plt.xlabel('Date')
    plt.tight_layout()
    
    
def _plot_new_cases_county(df):
    num_counties = len(list(set(df['county'])))
    
    if num_counties > 27:
        num_col = int(num_counties/27) + 1
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'new_cases', hue='county', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('New Cases')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_new_deaths_state(df):
    num_states = len(list(set(df['state'])))
    
    if num_states > 27:
        num_col = 2
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'new_deaths', hue='state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('New Deaths')
    plt.xlabel('Date')
    plt.tight_layout()
    
    
def _plot_new_deaths_county(df):
    num_counties = len(list(set(df['county'])))
    
    if num_counties > 27:
        num_col = int(num_counties/27) + 1
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'new_deaths', hue='county', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.ylabel('New Deaths')
    plt.xlabel('Date')
    plt.tight_layout()


def plot_state_cases_vs_deaths(df):
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
    
    
def _plot_state_moving_average(df):
    
    plt.figure(figsize=(12, 6))
    sns.lineplot('date', 'new_cases', data=df, color="b", legend=None)
    plt.ylabel('Cases')
    plt.xticks(rotation=90)
    
    sns.lineplot('date', 'moving_ave', data=df, color="r", legend=None)
    
    
    blue_line = mlines.Line2D([], [], color='blue', marker=None, label='New Cases')
    red_line = mlines.Line2D([], [], color='red', marker=None, label="Fitted")
    plt.legend(loc='upper left', handles=[blue_line, red_line])
    
    plt.ylabel('Fitted')
    plt.xlabel('Date')
    plt.tight_layout()
    
    
def _plot_county_moving_average(df):
    
    plt.figure(figsize=(12, 6))
    sns.lineplot('date', 'new_cases', data=df, color="b", legend=None)
    plt.ylabel('Cases')
    plt.xticks(rotation=90)
    
    sns.lineplot('date', 'moving_ave', data=df, color="r", legend=None)
    
    
    blue_line = mlines.Line2D([], [], color='blue', marker=None, label='New Cases')
    red_line = mlines.Line2D([], [], color='red', marker=None, label="Fitted")
    plt.legend(loc='upper left', handles=[blue_line, red_line])
    
    plt.ylabel('Fitted')
    plt.xlabel('Date')
    plt.tight_layout()
    
    

    
    
    
    

    