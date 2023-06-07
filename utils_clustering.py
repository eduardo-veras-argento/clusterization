# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:52:02 2020

@author: Pedro R. Suanno
"""
#%% Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import datetime
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import statistics as stats
import seaborn as sns
import time
import json
from cycler import cycler
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib import cm
#from colorspacious import cspace_converter
from collections import OrderedDict
from factor_analyzer import FactorAnalyzer, Rotator

from constants_clustering import *
#%% Functions definitions

def calculate_avg(df):
    """
    Function to avg a feature
    
    df.iloc[0] : nb of matchs
    df.iloc[1] : feature to avg
    """
    return df.iloc[1:]/df.iloc[0]

def calculate_accuracy(df):
    """
    Function to calculate accuracy of a certain feature
    
    df.iloc[0] : feature value tried
    df.iloc[1] : feature value lost
    """
    if df.iloc[0] == 0:
        return 0
    return 1 - df.iloc[1]/df.iloc[0]

def get_info_from_id(id_player,info):
    """
    Function to get info from info file using player info
    
    info : info name
    id_player : player id
    """
#    print(id_player)
    mask = df_players_info['id'] == id_player
#    print(df_players_info[mask][info])
    if isinstance(df_players_info[mask][info].values, list):
        print('Here boy {}'.format(df_players_info[mask][info].values))
    return df_players_info[mask][info].values[0]

def get_team_from_id(id_player,info):
    """
    Function to get info from info file using player info
    
    info : info name
    id_player : player id
    """
#    print(id_player)
    mask = df_team_players['player_id'] == id_player
#    print(df_players_info[mask][info])
    if isinstance(df_team_players[mask][info].values, list):
        print('Here boy {}'.format(df_team_players[mask][info].values))
    return df_team_players[mask][info].values[0]

def create_series_PT(s_p_row):
    """
    Function to create a series with total time played 
    
    s_p_row : row containing season_id and player_id
    """
#    print('s_p_row : {}'.format(s_p_row))
    s_id = s_p_row['season_id']
    p_id = s_p_row['player_id']
    s_idx = df_players_stats.loc[:,'season_id'] == s_id
#    print('s_idx : {}'.format(s_idx))
    p_idx = df_players_stats.loc[:,'athlete_id'] == p_id
    s_p_idx = np.logical_and(s_idx,p_idx)
#    print('sum spidsx : {}'.format(s_p_idx.sum()))
    total_minutes_pl = df_players_stats.loc[s_p_idx,'played_minutes']
#    print('total_minutes_pl : {}'.format(total_minutes_pl))
    return total_minutes_pl.values[0]
    
def calculate_p_i(s_p_row):
    """
    Function to calculate p_i for a pair of players and season
    
    s_p_row : row containing season_id and player_id
    """
    p_i = s_p_row['played_minutes']/(4 * s_p_row['total_minutes'])
    return p_i

def calculate_entropy(row_player):
    """
    Function to calculate the entropy for a player and season
    
    s_p_row : row containing season_id and player_id informations regarding time played
    """
    p_i = row_player['p_i'].values
    log_p_i = np.log2(p_i)
    LE = - np.dot(p_i,log_p_i)
    
    
    return LE


# Define rfm_level function
def tjl_level(df):
    """
    Function to calculate the TJL level
    """
    if df['TJL_score'] >= 9:
        return 'Omnipresent'
    elif ((df['TJL_score'] >= 8) and (df['TJL_score'] < 9)):
        return 'Highly tested'
    elif ((df['TJL_score'] >= 7) and (df['TJL_score'] < 8)):
        return 'Reliable'
    elif ((df['TJL_score'] >= 6) and (df['TJL_score'] < 7)):
        return 'Potential'
    elif ((df['TJL_score'] >= 5) and (df['TJL_score'] < 6)):
        return 'Promising'
    elif ((df['TJL_score'] >= 4) and (df['TJL_score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Biased'