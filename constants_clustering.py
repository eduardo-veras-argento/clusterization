# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:11:04 2020

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

#%% load files 
df_players_stats = pd.read_csv("Player_Stats_by_Season2022.csv",header = 0, decimal = ',',encoding='utf-8')
df_players_stats = df_players_stats[['season_id','athlete_id','played_matches','total_points','lost_points','tried_points','two_points'
                                    ,'tried_two_points','lost_two_points','three_points','tried_three_points','lost_three_points',
                                    'free_throw_points','tried_free_throw_points','lost_free_throw_points','dunk_points',
                                    'tried_dunk_points','lost_dunk_points','tap_points','tried_tap_points','lost_tap_points',
                                    'total_rebounds','offensive_rebounds','defensive_rebounds','steals','total_assists','total_throws'
                                    ,'tried_throws','correct_throws','wrong_throws','two_points_throws_total',
                                    'correct_two_points_throws','wrong_two_points_throws','three_points_throws_total',
                                    'correct_three_points_throws','wrong_three_points_throws','total_free_throws','correct_free_throws'
                                    ,'wrong_free_throws','total_of_dunks','correct_dunks','wrong_dunks','total_of_taps','correct_taps'
                                    ,'wrong_taps','total_of_rebound_shots','correct_rebound_shots','offensive_fouls',
                                    'disqualifying_fouls','technical_fouls','unsportsmanlike_fouls','commited_fouls','received_fouls'
                                    ,'total_fouls','total_of_violations','twenty_four_seconds_violations','eight_seconds_violations',
                                    'five_seconds_violations','three_seconds_violations','conducting_violations','field_back_violations'
                                    ,'interception_violations','out_field_violations','walk_violations','total_of_blocks',
                                    'total_of_errors','double_double_total','triple_double_total','bench_points','played_minutes',
                                    'efficiency','plus_minus']].copy()
#df_players_stats = pd.read_csv('nbb_athletes_stats_by_season_2.csv", error_bad_lines=False, header = 0, delimiter = ',', decimal = ',',encoding='utf-8')
#df_players_stats['played_minutes'] = df_players_stats['played_minutes'].str.replace('.','') 
# take a look into played minutes

#df_players_info = pd.read_csv("nbb-players-new1.csv", error_bad_lines=False, header = 0, delimiter = ',',decimal = ',',encoding='utf-8')
df_players_info = pd.read_csv("Player_Profile2022.csv",header = 0,decimal = ',',encoding='utf-8')

#df_minutes_players = pd.read_csv("player_time.csv", error_bad_lines=False, header = 0, delimiter = ',', decimal = ',', encoding = "ISO-8859-1")
#df_minutes_players['played_minutes'] = df_minutes_players['played_minutes'].astype(float)
df_minutes_players = pd.read_csv("player_time2.csv",header = 0, delimiter = ',', decimal = '.',encoding='utf-8')

df_team_players = pd.read_csv("player_teams.csv", header = 0, delimiter = ',', decimal = ',',encoding='utf-8')
#%% Constants
FLAG_outlier = True
FLAG_entropy = True
FLAG_stand = False
FLAG_norm = True
if FLAG_stand:
    FLAG_norm = True
FLAG_dot = True
FLAG_write = True
FLAG_adjust_clusters = True
var_to_avg = 'played_minutes'
minutes_min_mask = 100
threshold_outliers = 4
#original_stats_to_consider = ["total_points","lost_points","tried_points",\
#                              "two_points","tried_two_points","lost_two_points",\
#                              "three_points","tried_three_points","lost_three_points",\
#                              "free_throw_points","tried_free_throw_points",
#                              "lost_free_throw_points","tap_points","tried_tap_points",\
#                              "lost_tap_points","total_rebounds","offensive_rebounds",\
#                              "defensive_rebounds","steals","total_assists",\
#                              "total_throws","tried_throws","correct_throws",\
#                              "wrong_throws","two_points_throws_total",\
#                              "correct_two_points_throws","wrong_two_points_throws",\
#                              "three_points_throws_total","correct_three_points_throws",\
#                              "wrong_three_points_throws","total_free_throws",\
#                              "correct_free_throws","wrong_free_throws","total_of_dunks",\
#                              "correct_dunks","wrong_dunks","total_of_taps",\
#                              "correct_taps","wrong_taps","total_of_rebound_shots",\
#                              "correct_rebound_shots","offensive_fouls","disqualifying_fouls",\
#                              "technical_fouls","unsportsmanlike_fouls","commited_fouls",\
#                              "received_fouls","total_fouls","total_of_violations",
#                              "twenty_four_seconds_violations","eight_seconds_violations",\
#                              "five_seconds_violations","three_seconds_violations",\
#                              "conducting_violations","field_back_violations",\
#                              "interception_violations","out_field_violations",\
#                              "walk_violations","total_of_blocks","total_of_errors",\
#                              "double_double_total","triple_double_total","bench_points"]

original_stats_to_consider = ["total_points","tried_points",\
                              "two_points","tried_two_points",\
                              "three_points","tried_three_points",\
                              "free_throw_points","tried_free_throw_points",
                              "tap_points","tried_tap_points",\
                              "total_rebounds","offensive_rebounds",\
                              "defensive_rebounds","steals","total_assists",\
                              "total_of_dunks",\
                              "correct_dunks",\
                              "total_of_rebound_shots",\
                              "correct_rebound_shots","offensive_fouls","disqualifying_fouls",\
                              "technical_fouls","unsportsmanlike_fouls","commited_fouls",\
                              "received_fouls","total_fouls","total_of_violations",
                              "twenty_four_seconds_violations","eight_seconds_violations",\
                              "five_seconds_violations","three_seconds_violations",\
                              "conducting_violations","field_back_violations",\
                              "interception_violations","out_field_violations",\
                              "walk_violations","total_of_blocks","total_of_errors",\
                              "double_double_total","triple_double_total","bench_points"]

#without thrieds
#original_stats_to_consider = ["total_points",\
#                              "two_points",\
#                              "three_points",\
#                              "free_throw_points",
#                              "tap_points","tried_tap_points",\
#                              "total_rebounds","offensive_rebounds",\
#                              "defensive_rebounds","steals","total_assists",\
#                              "total_of_dunks",\
#                              "correct_dunks",\
#                              "total_of_rebound_shots",\
#                              "correct_rebound_shots","offensive_fouls","disqualifying_fouls",\
#                              "technical_fouls","unsportsmanlike_fouls","commited_fouls",\
#                              "received_fouls","total_fouls","total_of_violations",
#                              "twenty_four_seconds_violations","eight_seconds_violations",\
#                              "five_seconds_violations","three_seconds_violations",\
#                              "conducting_violations","field_back_violations",\
#                              "interception_violations","out_field_violations",\
#                              "walk_violations","total_of_blocks","total_of_errors",\
#                              "double_double_total","triple_double_total","bench_points"]

#original_stats_to_consider = ["total_points","lost_points","tried_points",\
#                              "two_points","tried_two_points","lost_two_points",\
#                              "three_points","tried_three_points","lost_three_points",\
#                              "free_throw_points","tried_free_throw_points",
#                              "lost_free_throw_points","tap_points","tried_tap_points",\
#                              "lost_tap_points","total_rebounds","offensive_rebounds",\
#                              "defensive_rebounds","steals","total_assists",\
#                              "total_throws","tried_throws","correct_throws",\
#                              "wrong_throws","two_points_throws_total",\
#                              "correct_two_points_throws","wrong_two_points_throws",\
#                              "three_points_throws_total","correct_three_points_throws",\
#                              "wrong_three_points_throws","total_free_throws",\
#                              "correct_free_throws","wrong_free_throws","total_of_dunks",\
#                              "correct_dunks","wrong_dunks","total_of_taps",\
#                              "correct_taps","wrong_taps","total_of_rebound_shots",\
#                              "correct_rebound_shots","offensive_fouls","disqualifying_fouls",\
#                              "technical_fouls","unsportsmanlike_fouls","commited_fouls",\
#                              "received_fouls","total_fouls","total_of_violations",
#                              "twenty_four_seconds_violations","eight_seconds_violations",\
#                              "five_seconds_violations","three_seconds_violations",\
#                              "conducting_violations","field_back_violations",\
#                              "interception_violations","out_field_violations",\
#                              "walk_violations","total_of_blocks","total_of_errors",\
#                              "double_double_total","triple_double_total","bench_points",\
#                              "efficiency","plus_minus"]

names_positions = {1 : 'Armador', 2 : 'Ala' , 3 : 'Pivo', 4 : 'Ala/Arm.', 5 : 'Ala/Piv.'}