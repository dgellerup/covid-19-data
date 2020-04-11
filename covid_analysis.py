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
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import us
from zipfile import ZipFile


def get_census_api_key():
    keyfile_dir = json.load(open('api_keypath.json')).get('api_keypath')
    path_to_keyfile = os.path.join(keyfile_dir, 'keys.json')
        
    return json.load(open(path_to_keyfile)).get('us_census')


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
    
    county_data.dropna(inplace=True)
    
    county_data['fips'] = county_data['fips'].astype(int)
    
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


def five_day_moving_average(iterable: Iterable) -> List[float]:
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
    
    
def nine_day_moving_average(iterable: Iterable) -> List[float]:
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


def fit_curve(iterable: Iterable, deg: int) -> List[int]:
    "This more or less does not work but I want to leave it here for future ref."
    x = range(0, len(iterable))
    y = iterable
    
    z = np.polyfit(x, y, deg)
    
    p = np.poly1d(z)
    
    fit = [int(p(f)) for f in x]
    
    return fit


def plot_cases(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        _plot_county_cases(df)
    else:
        _plot_state_cases(df)
        
        
def plot_deaths(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        _plot_county_deaths(df)
    else:
        _plot_state_deaths(df)


def _plot_state_cases(df: pd.DataFrame) -> None:
    
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
    

def _plot_state_deaths(df: pd.DataFrame) -> None:
    
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
    

def _plot_county_cases(df: pd.DataFrame) -> None:
    
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
    

def _plot_county_deaths(df: pd.DataFrame) -> None:
    
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


def plot_new_cases(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        _plot_new_cases_county(df)
    else:
        _plot_new_cases_state(df)
        
        
def plot_new_deaths(df: pd.DataFrame) -> None:
    if "county, state" in df.columns:
        _plot_new_deaths_county(df)
    else:
        _plot_new_deaths_state(df)
        
    
def _plot_new_cases_state(df: pd.DataFrame) -> None:
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
    
    
def _plot_new_cases_county(df: pd.DataFrame) -> None:
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
    

def _plot_new_deaths_state(df: pd.DataFrame) -> None:
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
    
    
def _plot_new_deaths_county(df: pd.DataFrame) -> None:
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
    
    
def _plot_state_moving_average(df: pd.DataFrame) -> None:
    
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
    
    
def _plot_county_moving_average(df: pd.DataFrame) -> None:
    
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
    
    for date in list(set(state_df['date'])):
        
        state_from_date = state_df[state_df['date'] == date]
        
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
        ax.set_title(f"Cases by County {date}",
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
        
        fig.savefig(os.path.join(plotpath, f'counties_{date}.png'), dpi=300)
        
        plt.close(fig)
        
        images.append(os.path.join(plotpath, f'counties_{date}.png'))
        
    images = sorted(images)
    
    gif_images = [imageio.imread(x) for x in images]
    
    imageio.mimsave(f'plots/{state}.gif', gif_images, duration=0.5)
    
    
def make_nice_wi_gif():
    
    state_data = load_state_data()
    state_data = state_data[state_data['state'] == 'Wisconsin']
    wi_state = get_data_since_date(state_data, '2020-03-01')
    
    county_data = load_county_data()
    state_df = county_data[county_data['state'] == 'Wisconsin']
            
    state_df['short_fips'] = state_df['fips'].apply(lambda x: int(str(int(x))[2:]))
    
    state_df = get_data_since_date(state_df, '2020-03-01')
    
    vmin, vmax = 0, state_df['cases'].max()
        
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
    
    for date in list(set(state_df['date'])):
        
        wi_state['era'] = wi_state['date'].apply(lambda x: 'old' if x <= date else 'new')
        
        winow = state_df[state_df['date'] == date]

        merged = map_df.set_index("COUNTY_FIP").join(winow.set_index('short_fips'))
        
        #merged['cases'] = merged['cases'].apply(lambda x: 0 if pd.isnull(x) else x)
        
        variable = 'cases'
        
        fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(9, 9))
        
        ax0.set_aspect('equal')
        
        merged.plot(column=variable,
                    cmap='viridis',
                    linewidth=0.8,
                    ax=ax0,
                    edgecolor='0.8',
                    vmin=vmin,
                    vmax=vmax,
                    missing_kwds={"color": "lightgrey"},
                    figsize=(6, 2))
        
        ax0.axis('off')
        
        # add a title
        ax0.set_title(f"Cases by County {date}",
                     fontdict={'fontsize': '18', 'fontweight' : '3'})
        
        # create an annotation for the data source
        ax0.annotate('Source: The New York Times, 2020',
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
    
    gif_images = [imageio.imread(x) for x in images]
    
    imageio.mimsave('plots/wisconsin.gif', gif_images, duration=0.5)
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    