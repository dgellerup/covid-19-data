#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:22:10 2020

@author: dgelleru
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def _plot_national_cases(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'cases', data=df)
    
    plt.xticks(size=6, rotation=90)
    plt.title('National Cases')
    plt.ylabel('Cases')
    plt.xlabel('Date')
    plt.tight_layout()


def _plot_national_deaths(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'deaths', data=df)
    
    plt.xticks(size=6, rotation=90)
    plt.title("National Deaths")
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    plt.tight_layout()
    

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
    plt.title('State Cases')
    plt.ylabel('Cases')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig('plots/plot.png', dpi=300)
    

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
    plt.title('State Deaths')
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
    plt.title('County Cases')
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
    plt.title('County Deaths')
    plt.ylabel('Deaths')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_cases_per_capita_national(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'cases_per_100k', data=df)
    
    plt.xticks(size=6, rotation=90)
    plt.title('National Cases per 100k People')
    plt.ylabel('Cases/100k')
    plt.xlabel('Date')
    plt.tight_layout()


def _plot_deaths_per_capita_national(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'deaths_per_100k', data=df)
    
    plt.xticks(size=6, rotation=90)
    plt.title("National Deaths per 100K People")
    plt.ylabel('Deaths/100k')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_cases_per_capita_state(df: pd.DataFrame) -> None:
    
    num_states = len(list(set(df['state'])))
    
    if num_states > 27:
        num_col = 2
    else:
        num_col = 1
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'cases_per_100k', hue='state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.title('State Cases per 100k People')
    plt.ylabel('Cases/100k')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_deaths_per_capita_state(df: pd.DataFrame) -> None:
    
    num_states = len(list(set(df['state'])))
    
    if num_states > 27:
        num_col = 2
    else:
        num_col = 1
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'deaths_per_100k', hue='state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.title('State Deaths per 100k People')
    plt.ylabel('Deaths/100k')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_cases_per_capita_county(df: pd.DataFrame) -> None:
    
    num_counties = len(list(set(df['county'])))
    
    if num_counties > 27:
        num_col = int(num_counties/27) + 1
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'cases_per_10k', hue='county, state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.title('County Cases per 10k People')
    plt.ylabel('Cases/10k')
    plt.xlabel('Date')
    plt.tight_layout()
    

def _plot_deaths_per_capita_county(df: pd.DataFrame) -> None:
    
    num_counties = len(list(set(df['county'])))
    
    if num_counties > 27:
        num_col = int(num_counties/27) + 1
    else:
        num_col = 1
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'deaths_per_10k', hue='county, state', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xticks(size=6, rotation=90)
    plt.title('County Deaths per 10k People')
    plt.ylabel('Deaths/10k')
    plt.xlabel('Date')
        
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('plots/deadly_counties.png', dpi=300)
    

def _plot_new_cases_national(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'new_cases', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6})
    plt.xticks(size=6, rotation=90)
    plt.title('New Cases: National')
    plt.ylabel('New Cases')
    plt.xlabel('Date')
    plt.tight_layout()
    
    
def _plot_new_deaths_national(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    
    sns.lineplot('date', 'new_deaths', data=df)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6})
    plt.xticks(size=6, rotation=90)
    plt.title('New Deaths: National')
    plt.ylabel('New Deaths')
    plt.xlabel('Date')
    plt.tight_layout()
    
    
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
    plt.title('New Cases: State')
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
    plt.title('New Deaths: State')
    plt.ylabel('New Deaths')
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
    plt.title('New Cases: County')
    plt.ylabel('New Cases')
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
    plt.title('New Deaths: County')
    plt.ylabel('New Deaths')
    plt.xlabel('Date')
    plt.tight_layout()
        
        
def _plot_national_moving_average(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(12, 6))
    sns.lineplot('date', 'moving_ave', data=df)
    plt.ylabel('Moving Average')
    plt.xticks(rotation=90)
    
    plt.xlabel('Date')
    plt.tight_layout()
    
        
def _plot_state_moving_average(df: pd.DataFrame) -> None:
    num_states = len(list(set(df['state'])))
    
    if num_states > 27:
        num_col = 2
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    sns.lineplot('date', 'moving_ave', hue='state', data=df)
    plt.ylabel('Moving Average')
    plt.xticks(rotation=90)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xlabel('Date')
    plt.tight_layout()
    
    
def _plot_county_moving_average(df: pd.DataFrame) -> None:
    num_counties = len(list(set(df['county'])))
    
    if num_counties > 27:
        num_col = int(num_counties/27) + 1
    else:
        num_col = 1
        
    plt.figure(figsize=(12, 6))
    sns.lineplot('date', 'moving_ave', hue='county', data=df)
    plt.ylabel('Moving Average')
    plt.xticks(rotation=90)
    
    plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':6}, ncol=num_col)
    plt.xlabel('Date')
    plt.tight_layout()