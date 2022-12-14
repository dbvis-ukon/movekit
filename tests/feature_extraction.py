import os
import unittest
from pandas import Timestamp
import pandas as pd
import numpy as np
import math
import pickle
from pandas.testing import assert_frame_equal
from tqdm import tqdm


from src.movekit.feature_extraction import *



# Required datasets for testing:
dict_groups = {
		312: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 312, 1: 312, 2: 312},
		'x': {0: 405.29, 1: 405.31, 2: 405.31},
		 'y': {0: 417.76, 1: 417.37, 2: 417.07}, 'distance': {0: None, 1: None, 2: None},
		 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
		 'direction': {0: None, 1: None, 2: None}, 'stopped': {0: None, 1: None, 2: None}}),
		511: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 511, 1: 511, 2: 511},
		'x': {0: 369.99, 1: 370.01, 2: 370.01},
		 'y': {0: 428.78, 1: 428.82, 2: 428.85}, 'distance': {0: None, 1: None, 2: None},
		 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
		 'direction': {0: None, 1: None, 2: None}, 'stopped': {0: None, 1: None, 2: None}}),
		607: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 607, 1: 607, 2: 607},
		'x': {0: 390.33, 1: 390.25, 2: 390.17},
		 'y': {0: 405.89, 1: 405.89, 2: 405.88}, 'distance': {0: None, 1: None, 2: None},
		 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
		 'direction': {0: None, 1: None, 2: None}, 'stopped': {0: None, 1: None, 2: None}}),
		811: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 811, 1: 811, 2: 811},
		'x': {0: 445.15, 1: 445.48, 2: 445.77},
		 'y': {0: 411.94, 1: 412.26, 2: 412.61}, 'distance': {0: None, 1: None, 2: None},
		 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
		 'direction': {0: None, 1: None, 2: None}, 'stopped': {0: None, 1: None, 2: None}}),
		905: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 905, 1: 905, 2: 905},
		'x': {0: 366.06, 1: 365.86, 2: 365.7},
		 'y': {0: 451.76, 1: 451.76, 2: 451.76}, 'distance': {0: None, 1: None, 2: None},
		 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
		 'direction': {0: None, 1: None, 2: None}, 'stopped': {0: None, 1: None, 2: None}})}

regroups = {
	'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3,
										  12: 1, 13: 2, 14: 3},
					'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607,
											   9: 811, 10: 811, 11: 811, 12: 905, 13: 905, 14: 905},
					'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33,
									   7: 390.25, 8: 390.17, 9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86,
									   14: 365.7},
					'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89,
									   7: 405.89, 8: 405.88, 9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76,
									   14: 451.76},
					'distance': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None,
											  8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None},
					'average_speed': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None,6: None,7: None,
												   8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None},
					'average_acceleration': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
														  7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13:
															  None, 14: None},
					'direction': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None,
											   8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None},
					'stopped': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None,
											 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None}}

euclidean = {
			'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3,
					  14: 3},
			 'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905,
						   10: 312, 11: 511, 12: 607, 13: 811, 14: 905},
			 'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01, 7: 390.25,
				   8: 445.48, 9: 365.86, 10: 405.31, 11: 370.01, 12: 390.17, 13: 445.77, 14: 365.7},
			 'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82, 7: 405.89,
				   8: 412.26, 9: 451.76, 10: 417.07, 11: 428.85, 12: 405.88, 13: 412.61, 14: 451.76},
			 "312": {0: 0.0, 1: 36.980135, 2: 19.097081, 3: 40.282651, 4: 51.913321, 5: 0.0, 6: 37.110544, 7: 18.936578, 8: 40.493716,
					 9: 52.335214, 10: 0.0, 11: 37.213686, 12: 18.826463, 13: 40.705076, 14: 52.653093 },
			 "511": {0: 36.980135, 1: 0.0, 2: 30.621360, 3: 77.023446, 4: 23.313629, 5: 37.110544, 6: 0.0, 7: 30.585004, 8: 77.265481,
					 9: 23.312359, 10: 37.213686, 11: 0.0, 12: 30.562174, 13: 77.481063 , 14: 23.311890},
			 "607": {0: 19.097081 , 1: 30.621360 , 2: 0.0, 3: 55.152832, 4: 51.894988 , 5: 18.936578, 6: 30.585004, 7: 0.0, 8: 55.596131,
					 9: 51.951218, 10: 18.826463, 11: 30.562174, 12: 0.0, 13: 56.005829, 14: 51.997647},
			 "811": {0: 40.282651, 1: 77.023446, 2: 55.152832, 3: 0.0, 4: 88.548634, 5: 40.493716, 6: 77.265481, 7: 55.596131,
					 8: 0.0, 9: 88.879662, 10: 40.705076, 11: 77.481063, 12: 56.005829, 13: 0.0, 14: 89.128713},
			 "905": {0: 51.913321, 1: 23.313629, 2: 51.894988, 3: 88.548634, 4: 0.0, 5: 52.335214, 6: 23.312359, 7: 51.951218,
					 8: 88.879662, 9: 0.0, 10: 52.653093, 11: 23.311890, 12: 51.997647, 13: 89.128713, 14: 0.0}}

records = {
			"time": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
			"animal_id": [312, 511, 607, 811, 905, 511, 811, 312, 905, 607, 312, 511, 607, 811, 905],
			"x": [405.29, 369.99, 390.33, 445.15, 366.06, 370.01, 445.48, 405.31, 365.86, 390.25, 405.31, 370.01,
				  390.17, 445.77, 365.7],
			"y": [417.76, 428.78, 405.89, 411.94, 451.76, 428.82, 412.26, 417.37, 451.76, 405.89, 417.07, 428.85,
				  405.88,
				  412.61, 451.76]
			}


records_timestring = {
        "time": ["2020-03-27 11:57:07", "2020-03-27 11:57:07", "2020-03-27 11:57:07", "2020-03-27 11:57:07", "2020-03-27 11:57:07",
                 "2020-03-27 11:57:09", "2020-03-27 11:57:09", "2020-03-27 11:57:09", "2020-03-27 11:57:09", "2020-03-27 11:57:09", "2020-03-27 11:57:11", "2020-03-27 11:57:11", "2020-03-27 11:57:11", "2020-03-27 11:57:11", "2020-03-27 11:57:11"],
        "animal_id": [312, 511, 607, 811, 905, 511, 811, 312, 905, 607, 312, 511, 607, 811, 905],
        "x": [405.29, 369.99, 390.33, 445.15, 366.06, 370.01, 445.48, 405.31, 365.86, 390.25, 405.31, 370.01,
              390.17, 445.77, 365.7],
        "y": [417.76, 428.78, 405.89, 411.94, 451.76, 428.82, 412.26, 417.37, 451.76, 405.89, 417.07, 428.85,
              405.88,
              412.61, 451.76]
        }

medoids = {
	'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3}, 'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905, 10: 312, 11: 511, 12: 607, 13: 811, 14: 905}, 'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01, 7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31, 11: 370.01, 12: 390.17, 13: 445.77, 14: 365.7}, 'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82, 7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07, 11: 428.85, 12: 405.88, 13: 412.61, 14: 451.76}, 'x_centroid': {0: 395.364, 1: 395.364, 2: 395.364, 3: 395.364, 4: 395.364, 5: 395.382, 6: 395.382, 7: 395.382, 8: 395.382, 9: 395.382, 10: 395.392, 11: 395.392, 12: 395.392, 13: 395.392, 14: 395.392}, 'y_centroid': {0: 423.226, 1: 423.226, 2: 423.226, 3: 423.226, 4: 423.226, 5: 423.22, 6: 423.22, 7: 423.22, 8: 423.22, 9: 423.22, 10: 423.234, 11: 423.234, 12: 423.234, 13: 423.234, 14: 423.234}, 'medoid': {0: 312, 1: 312, 2: 312, 3: 312, 4: 312, 5: 312, 6: 312, 7: 312, 8: 312, 9: 312, 10: 312, 11: 312, 12: 312, 13: 312, 14: 312}, 'distance_to_centroid': {0: 11.331, 1: 25.975, 2: 18.052, 3: 51.049, 4: 40.901, 5: 11.523, 6: 25.983, 7: 18.074, 8: 51.283, 9: 41.062, 10: 11.677, 11: 25.996, 12: 18.123, 13: 51.486, 14: 41.175}}

distdic = {
			312: pd.DataFrame(
			{'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 312, 1: 312, 2: 312}, 'x': {0: 405.29, 1: 405.31, 2: 405.31},
			 'y': {0: 417.76, 1: 417.37, 2: 417.07}, 'distance': {0: 0.0, 1: 0.39051248379531817,
																  2: 0.30000000000001137},
			 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
			 'direction': {0: None, 1: -87.0643265535814, 2: -90.0}, 'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame(
				{'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 511, 1: 511, 2: 511},
				 'x': {0: 369.99, 1: 370.01, 2: 370.01}, 'y': {0: 428.78, 1: 428.82, 2: 428.85},
				 'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
				 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
				 'direction': {0: None, 1: 63.434948822954574, 2: 90.0}, 'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame(
				{'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 607, 1: 607, 2: 607},
				 'x': {0: 390.33, 1: 390.25, 2: 390.17}, 'y': {0: 405.89, 1: 405.89, 2: 405.88},
				 'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
				 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
				 'direction': {0: None, 1: 180.0, 2: -172.87498365110324}, 'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame(
				{'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 811, 1: 811, 2: 811},
				 'x': {0: 445.15, 1: 445.48, 2: 445.77}, 'y': {0: 411.94, 1: 412.26, 2: 412.61},
				 'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
				 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
				 'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
				 'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame(
				{'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 905, 1: 905, 2: 905},
				 'x': {0: 366.06, 1: 365.86, 2: 365.7}, 'y': {0: 451.76, 1: 451.76, 2: 451.76},
				 'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
				 'average_speed': {0: None, 1: None, 2: None}, 'average_acceleration': {0: None, 1: None, 2: None},
				 'direction': {0: None, 1: 180.0, 2: 180.0}, 'stopped': {0: None, 1: None, 2: None}})}

avspeed = {
			312: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 312, 1: 312, 2: 312},
							'x': {0: 405.29, 1: 405.31, 2: 405.31}, 'y': {0: 417.76, 1: 417.37, 2: 417.07},
							'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
							'average_speed': {0: 0.195256, 1: 0.345256, 2: 0.150000},
							'average_acceleration': {0: None, 1: None, 2: None},
							'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
							'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 511, 1: 511, 2: 511},
							   'x': {0: 369.99, 1: 370.01, 2: 370.01}, 'y': {0: 428.78, 1: 428.82, 2: 428.85},
							   'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
							   'average_speed': {0: 0.022361, 1: 0.037361, 2: 0.015000},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
							   'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 607, 1: 607, 2: 607},
							   'x': {0: 390.33, 1: 390.25, 2: 390.17}, 'y': {0: 405.89, 1: 405.89, 2: 405.88},
							   'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
							   'average_speed': {0: 0.040000, 1: 0.080311, 2: 0.040311},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
							   'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 811, 1: 811, 2: 811},
							   'x': {0: 445.15, 1: 445.48, 2: 445.77}, 'y': {0: 411.94, 1: 412.26, 2: 412.61},
							   'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
							   'average_speed': {0: 0.229837, 1: 0.457103, 2: 0.227266},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
							   'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 905, 1: 905, 2: 905},
							   'x': {0: 366.06, 1: 365.86, 2: 365.7}, 'y': {0: 451.76, 1: 451.76, 2: 451.76},
							   'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
							   'average_speed': {0: 0.10, 1: 0.18, 2: 0.08},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 180.0, 2: 180.0}, 'stopped': {0: None, 1: None, 2: None}})}

avaccel = {
			312 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 312, 1: 312, 2: 312},
					   			'x': {0: 405.29, 1: 405.31, 2: 405.31},
								'y': {0: 417.76, 1: 417.37, 2: 417.07},
								'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
								'average_speed': {0: 0.195256, 1: 0.345256, 2: 0.150000},
								'average_acceleration': {0:  0.075000, 1: -0.022628, 2: -0.097628},
								'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
								'stopped': {0: None, 1: None, 2: None}} ),
			511 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
								 'animal_id': {0: 511, 1: 511, 2: 511}, 'x': {0: 369.99,
																			  1: 370.01, 2: 370.01},
								 'y': {0: 428.78, 1: 428.82, 2: 428.85},
								 'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								 'average_speed': {0: 0.022361,1: 0.037361, 2: 0.015000},
								 'average_acceleration': {0: 0.007499999999999998, 1: -0.0036804999999999997, 2: -0.0111805},
								 'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
								 'stopped': {0: None, 1: None, 2: None}} ),
			607 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 607, 1: 607, 2: 607},
								'x': {0: 390.33, 1: 390.25, 2: 390.17},
								'y': {0: 405.89, 1: 405.89, 2: 405.88},
								'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								'average_speed': {0: 0.040000, 1: 0.080311, 2: 0.040311},
								'average_acceleration': {0: 0.0201555, 1: 0.0001554999999999994, 2: -0.019999999999999997},
								'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
								'stopped': {0: None, 1: None, 2: None}} ),
			811 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 811, 1: 811, 2: 811},
								'x': {0: 445.15, 1: 445.48, 2: 445.77},
								'y': {0: 411.94, 1: 412.26, 2: 412.61},
								'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
								'average_speed': {0: 0.229837, 1: 0.457103, 2: 0.227266},
								'average_acceleration': {0: 0.113633, 1: -0.0012855000000000089, 2: -0.11491849999999998},
								'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
								'stopped': {0: None, 1: None, 2: None}} ),
			905 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 905, 1: 905, 2: 905},
								'x': {0: 366.06, 1: 365.86, 2: 365.7},
								'y': {0: 451.76, 1: 451.76, 2: 451.76},
								'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								'average_speed': {0: 0.10, 1: 0.18, 2: 0.08},
								'average_acceleration': {0: 0.04, 1: -0.01, 2: -0.05},
								'direction': {0: None, 1: 180.0, 2: 180.0},
								'stopped': {0: None, 1: None, 2: None}})}

stops = {
			312 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 312, 1: 312, 2: 312},
					   			'x': {0: 405.29, 1: 405.31, 2: 405.31},
								'y': {0: 417.76, 1: 417.37, 2: 417.07},
								'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
								'average_speed': {0: 0.195256, 1: 0.345256, 2: 0.150000},
								'average_acceleration': {0:  0.075000, 1: -0.022628, 2: -0.097628},
								'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
								'stopped': {0: 0, 1: 0, 2: 0}} ),
			511 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
								 'animal_id': {0: 511, 1: 511, 2: 511}, 'x': {0: 369.99,
																			  1: 370.01, 2: 370.01},
								 'y': {0: 428.78, 1: 428.82, 2: 428.85},
								 'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								 'average_speed': {0: 0.022361,1: 0.037361, 2: 0.015000},
								 'average_acceleration': {0: 0.007499999999999998, 1: -0.0036804999999999997, 2: -0.0111805},
								 'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
								 'stopped': {0: 1, 1: 1, 2: 1}} ),
			607 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 607, 1: 607, 2: 607},
								'x': {0: 390.33, 1: 390.25, 2: 390.17},
								'y': {0: 405.89, 1: 405.89, 2: 405.88},
								'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								'average_speed': {0: 0.040000, 1: 0.080311, 2: 0.040311},
								'average_acceleration': {0: 0.0201555, 1: 0.0001554999999999994, 2: -0.019999999999999997},
								'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
								'stopped': {0: 1, 1: 1, 2: 1}} ),
			811 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 811, 1: 811, 2: 811},
								'x': {0: 445.15, 1: 445.48, 2: 445.77},
								'y': {0: 411.94, 1: 412.26, 2: 412.61},
								'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
								'average_speed': {0: 0.229837, 1: 0.457103, 2: 0.227266},
								'average_acceleration': {0: 0.113633, 1: -0.0012855000000000089, 2: -0.11491849999999998},
								'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
								'stopped': {0: 0, 1: 0, 2: 0}} ),
			905 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 905, 1: 905, 2: 905},
								'x': {0: 366.06, 1: 365.86, 2: 365.7},
								'y': {0: 451.76, 1: 451.76, 2: 451.76},
								'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								'average_speed': {0: 0.10, 1: 0.18, 2: 0.08},
								'average_acceleration': {0: 0.04, 1: -0.01, 2: -0.05},
								'direction': {0: None, 1: 180.0, 2: 180.0},
								'stopped': {0: 1, 1: 0, 2: 1}})}


feats = {'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3, 12: 1, 13: 2, 14: 3},
		 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811, 11: 811,
					   12: 905, 13: 905, 14: 905},
		 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
			   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
		 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
			   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
		 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
					  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
		'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3), 3: (0.0, 0.0), 4: (0.02, 0.04), 5: (0.0, 0.03), 6: (0.0, 0.0), 7: (-0.08, 0.0), 8: (-0.08, -0.01),
					   9: (0.0, 0.0), 10: (0.33, 0.32), 11: (0.29, 0.35), 12: (0.0, 0.0), 13: (-0.2, 0.0), 14: (-0.16, 0.0)},
		'turning': {0: 0.000000, 1: 0.0, 2: 0.9987, 3: 0.000000, 4: 0.0, 5: 0.8944, 6: 0.000000, 7: 0.000000, 8: 0.9923, 9: 0.000000,
							   10: 0.0, 11: 0.9941, 12: 0.000000, 13: 0.000000, 14: 1.000000},
		 'average_speed': {0: 0.1953, 1: 0.3453, 2: 0.1500, 3: 0.0224, 4: 0.0374, 5: 0.0150, 6: 0.0400, 7: 0.0803, 8: 0.0403,
						   9: 0.2298, 10: 0.4571, 11: 0.2273, 12: 0.10, 13: 0.18, 14: 0.08},
		 'average_acceleration': {0: 0.0750, 1: -0.0226, 2: -0.0976, 3: 0.0075, 4: -0.0037, 5: -0.0112, 6: 0.0202, 7: 0.0002,
								  8: -0.0200, 9: 0.1136, 10: -0.0013, 11: -0.1149, 12: 0.04, 13: -0.01, 14: -0.05},
		 'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1}}

timestamp_feats = {'time': {0: Timestamp('2020-03-27 11:57:07'), 1: Timestamp('2020-03-27 11:57:09'),
							2: Timestamp('2020-03-27 11:57:11'), 3: Timestamp('2020-03-27 11:57:07'),
							4: Timestamp('2020-03-27 11:57:09'), 5: Timestamp('2020-03-27 11:57:11'),
							6: Timestamp('2020-03-27 11:57:07'), 7: Timestamp('2020-03-27 11:57:09'),
							8: Timestamp('2020-03-27 11:57:11'), 9: Timestamp('2020-03-27 11:57:07'),
							10: Timestamp('2020-03-27 11:57:09'), 11: Timestamp('2020-03-27 11:57:11'),
							12: Timestamp('2020-03-27 11:57:07'), 13: Timestamp('2020-03-27 11:57:09'),
							14: Timestamp('2020-03-27 11:57:11')},
		 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811, 11: 811,
					   12: 905, 13: 905, 14: 905},
		 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
			   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
		 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
			   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
		 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
					  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
		'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3), 3: (0.0, 0.0), 4: (0.02, 0.04), 5: (0.0, 0.03), 6: (0.0, 0.0), 7: (-0.08, 0.0), 8: (-0.08, -0.01),
					   9: (0.0, 0.0), 10: (0.33, 0.32), 11: (0.29, 0.35), 12: (0.0, 0.0), 13: (-0.2, 0.0), 14: (-0.16, 0.0)},
		'turning': {0: 0.000000, 1: 0.0, 2: 0.9987, 3: 0.000000, 4: 0.0, 5: 0.8944, 6: 0.000000, 7: 0.000000, 8: 0.9923, 9: 0.000000,
							   10: 0.0, 11: 0.9941, 12: 0.000000, 13: 0.000000, 14: 1.000000},
		 'average_speed': {0: 0.0976, 1: 0.1726, 2: 0.0750, 3: 0.0112, 4: 0.0187, 5: 0.0075, 6: 0.0200, 7: 0.0402, 8: 0.0202,
						   9: 0.1149, 10: 0.2286, 11: 0.1136, 12: 0.05, 13: 0.09, 14: 0.04},
		 'average_acceleration': {0: 0.0188, 1: -0.0057, 2: -0.0244, 3: 0.0019, 4: -0.0009, 5: -0.0028, 6: 0.005, 7: 0.000,
								  8: -0.005, 9: 0.0284, 10: -0.0003, 11: -0.0287, 12: 0.0100, 13: -0.0025, 14: -0.0125},
		 'stopped': {0: 1, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 0, 10: 0, 11: 0, 12: 1, 13: 1, 14: 1}}


ts_feat = {'average_speed__autocorrelation__lag_0': {312: 1.0, 511: 1.0, 607: 1.0, 811: 1.0, 905: 1.0},
		   'average_speed__autocorrelation__lag_1': {312: -0.9509, 511: -0.8949, 607: -1.0, 811: -0.9999, 905: -0.9643},
		   'average_speed__autocorrelation__lag_2': {312: 0.4018, 511: 0.2898, 607: 0.4999, 811: 0.4998, 905: 0.4286}}

ts_feats = {'x__mean_second_derivative_central':
				{312: -0.01, 511: -0.01, 607: 0.0, 811: -0.02, 905: 0.02},
			'x__median':
				{312: 405.31, 511: 370.01, 607: 390.25, 811: 445.48, 905: 365.86},
			'x__mean':
				{312: 405.3033, 511: 370.0033, 607: 390.25, 811: 445.4667, 905: 365.8733}}


class Test_Feature_Extraction(unittest.TestCase):
	'''
	Unit Tests for Feature Extraction
	'''

	def test_grouping_data(self):
		"""
		Testing grouping data function by animal ID for optimal tracking.
		:return:
		"""
		records = {
			"time": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
			"animal_id": [312, 511, 607, 811, 905, 511, 811, 312, 905, 607, 312, 511, 607, 811, 905],
			"x": [405.29, 369.99, 390.33, 445.15, 366.06, 370.01, 445.48, 405.31, 365.86, 390.25, 405.31, 370.01,
				  390.17, 445.77, 365.7],
			"y": [417.76, 428.78, 405.89, 411.94, 451.76, 428.82, 412.26, 417.37, 451.76, 405.89, 417.07, 428.85,
				  405.88, 412.61, 451.76]}

		dict_groups = {
			312: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 312, 1: 312, 2: 312},
								  'x': {0: 405.29, 1: 405.31, 2: 405.31},
								  'y': {0: 417.76, 1: 417.37, 2: 417.07},
								  'distance': {0: None, 1: None, 2: None},
								  'average_speed': {0: None, 1: None, 2: None},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: None, 2: None},
								  'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 511, 1: 511, 2: 511},
								  'x': {0: 369.99, 1: 370.01, 2: 370.01},
								  'y': {0: 428.78, 1: 428.82, 2: 428.85},
								  'distance': {0: None, 1: None, 2: None},
								  'average_speed': {0: None, 1: None, 2: None},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: None, 2: None},
								  'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 607, 1: 607, 2: 607},
								  'x': {0: 390.33, 1: 390.25, 2: 390.17},
								  'y': {0: 405.89, 1: 405.89, 2: 405.88},
								  'distance': {0: None, 1: None, 2: None},
								  'average_speed': {0: None, 1: None, 2: None},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: None, 2: None},
								  'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 811, 1: 811, 2: 811},
								  'x': {0: 445.15, 1: 445.48, 2: 445.77},
								  'y': {0: 411.94, 1: 412.26, 2: 412.61},
								  'distance': {0: None, 1: None, 2: None},
								  'average_speed': {0: None, 1: None, 2: None},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: None, 2: None},
								  'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 905, 1: 905, 2: 905},
								  'x': {0: 366.06, 1: 365.86, 2: 365.7},
								  'y': {0: 451.76, 1: 451.76, 2: 451.76},
								  'distance': {0: None, 1: None, 2: None},
								  'average_speed': {0: None, 1: None, 2: None},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: None, 2: None},
								  'stopped': {0: None, 1: None, 2: None}})}

		inp_data = records

		ref = dict_groups
		inp = pd.DataFrame(inp_data)
		case = grouping_data(inp)
		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i],check_dtype=False)


	def test_regrouping_data(self):

		dict_groups = {
			312: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 312, 1: 312, 2: 312},
				'x': {0: 405.29, 1: 405.31, 2: 405.31},
				'y': {0: 417.76, 1: 417.37, 2: 417.07},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 511, 1: 511, 2: 511},
				'x': {0: 369.99, 1: 370.01, 2: 370.01},
				'y': {0: 428.78, 1: 428.82, 2: 428.85},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 607, 1: 607, 2: 607},
				'x': {0: 390.33, 1: 390.25, 2: 390.17},
				'y': {0: 405.89, 1: 405.89, 2: 405.88},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 811, 1: 811, 2: 811},
				'x': {0: 445.15, 1: 445.48, 2: 445.77},
				'y': {0: 411.94, 1: 412.26, 2: 412.61},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 905, 1: 905, 2: 905},
				'x': {0: 366.06, 1: 365.86, 2: 365.7},
				'y': {0: 451.76, 1: 451.76, 2: 451.76},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}})}

		regroups = {
			'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3,
										  12: 1, 13: 2, 14: 3},
					'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607,
											   9: 811, 10: 811, 11: 811, 12: 905, 13: 905, 14: 905},
					'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33,
									   7: 390.25, 8: 390.17, 9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86,
									   14: 365.7},
					'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89,
									   7: 405.89, 8: 405.88, 9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76,
									   14: 451.76},
					'distance': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None,
											  8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None},
					'average_speed': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None,6: None,7: None,
												   8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None},
					'average_acceleration': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None,
														  7: None, 8: None, 9: None, 10: None, 11: None, 12: None, 13:
															  None, 14: None},
					'direction': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None,
											   8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None},
					'stopped': {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None,
											 8: None, 9: None, 10: None, 11: None, 12: None, 13: None, 14: None}}


		inp = dict_groups
		ref = pd.DataFrame(regroups)
		ref.sort_values(['time', 'animal_id'], ascending=True, inplace=True)
		ref.reset_index(drop=True, inplace=True)
		case = regrouping_data(inp)
		pd.testing.assert_frame_equal(ref, case, check_dtype=False)

	def test_eucledian_dist(self):  # test fails because in ref we use the old function with errors (f.e. distance is calculated with normalized values)
		"""
		Testing euclidean distance calculation. Result of function on records data set is compared with reference over
		all columns.
		Note: This test also includes the functions "compute_similarity()" and "similarity_computation()" since
		"euclidean_dist()" builds upon them.
		:return: Logical, if function returns expected result on given data frames.
		"""
		ref = pd.DataFrame(euclidean).sort_values(by=["time", "animal_id"]).reset_index(drop=True)
		inp = pd.DataFrame(records)
		case = euclidean_dist(inp).sort_values(by=["time", "animal_id"]).reset_index(drop=True).round(4)
		case = case.rename(columns = str)
		pd.testing.assert_frame_equal(ref, case, check_dtype=False)


	def test_compute_distance(self):
		distdic = {
			312: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 312, 1: 312, 2: 312},
				'x': {0: 405.29, 1: 405.31, 2: 405.31},
				'y': {0: 417.76, 1: 417.37, 2: 417.07},
				'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 511, 1: 511, 2: 511},
				'x': {0: 369.99, 1: 370.01, 2: 370.01},
				'y': {0: 428.78, 1: 428.82, 2: 428.85},
				'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 607, 1: 607, 2: 607},
				'x': {0: 390.33, 1: 390.25, 2: 390.17},
				'y': {0: 405.89, 1: 405.89, 2: 405.88},
				'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 811, 1: 811, 2: 811},
				'x': {0: 445.15, 1: 445.48, 2: 445.77},
				'y': {0: 411.94, 1: 412.26, 2: 412.61},
				'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 905, 1: 905, 2: 905},
				'x': {0: 366.06, 1: 365.86, 2: 365.7},
				'y': {0: 451.76, 1: 451.76, 2: 451.76},
				'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}})}
		dict_groups = {
			312: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 312, 1: 312, 2: 312},
				'x': {0: 405.29, 1: 405.31, 2: 405.31},
				'y': {0: 417.76, 1: 417.37, 2: 417.07},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 511, 1: 511, 2: 511},
				'x': {0: 369.99, 1: 370.01, 2: 370.01},
				'y': {0: 428.78, 1: 428.82, 2: 428.85},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 607, 1: 607, 2: 607},
				'x': {0: 390.33, 1: 390.25, 2: 390.17},
				'y': {0: 405.89, 1: 405.89, 2: 405.88},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 811, 1: 811, 2: 811},
				'x': {0: 445.15, 1: 445.48, 2: 445.77},
				'y': {0: 411.94, 1: 412.26, 2: 412.61},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 905, 1: 905, 2: 905},
				'x': {0: 366.06, 1: 365.86, 2: 365.7},
				'y': {0: 451.76, 1: 451.76, 2: 451.76},
				'distance': {0: None, 1: None, 2: None},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}})}

		inp = dict_groups
		ref = distdic
		case = compute_distance(inp)

		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i], check_dtype=False)

	def test_compute_direction(self):
		dirdic = {
			312: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 312, 1: 312, 2: 312},
				'x': {0: 405.29, 1: 405.31, 2: 405.31},
				'y': {0: 417.76, 1: 417.37, 2: 417.07},
				'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3)},
				'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 511, 1: 511, 2: 511},
				'x': {0: 369.99, 1: 370.01, 2: 370.01},
				'y': {0: 428.78, 1: 428.82, 2: 428.85},
				'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: (0.0, 0.0), 1: (0.02, 0.04), 2: (0.0, 0.03)},
				'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 607, 1: 607, 2: 607},
				'x': {0: 390.33, 1: 390.25, 2: 390.17},
				'y': {0: 405.89, 1: 405.89, 2: 405.88},
				'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: (0.0, 0.0), 1: (-0.08, 0.0), 2: (-0.08, -0.01)},
				'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 811, 1: 811, 2: 811},
				'x': {0: 445.15, 1: 445.48, 2: 445.77},
				'y': {0: 411.94, 1: 412.26, 2: 412.61},
				'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: (0.0, 0.0), 1: (0.33, 0.32), 2: (0.29, 0.35)},
				'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 905, 1: 905, 2: 905},
				'x': {0: 366.06, 1: 365.86, 2: 365.7},
				'y': {0: 451.76, 1: 451.76, 2: 451.76},
				'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: (0.0, 0.0), 1: (-0.2, 0.0), 2: (-0.16, 0.0)},
				'stopped': {0: None, 1: None, 2: None}})}

		dict_groups = {
			312: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 312, 1: 312, 2: 312},
				'x': {0: 405.29, 1: 405.31, 2: 405.31},
				'y': {0: 417.76, 1: 417.37, 2: 417.07},
				'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 511, 1: 511, 2: 511},
				'x': {0: 369.99, 1: 370.01, 2: 370.01},
				'y': {0: 428.78, 1: 428.82, 2: 428.85},
				'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 607, 1: 607, 2: 607},
				'x': {0: 390.33, 1: 390.25, 2: 390.17},
				'y': {0: 405.89, 1: 405.89, 2: 405.88},
				'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 811, 1: 811, 2: 811},
				'x': {0: 445.15, 1: 445.48, 2: 445.77},
				'y': {0: 411.94, 1: 412.26, 2: 412.61},
				'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 905, 1: 905, 2: 905},
				'x': {0: 366.06, 1: 365.86, 2: 365.7},
				'y': {0: 451.76, 1: 451.76, 2: 451.76},
				'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: None, 2: None},
				'stopped': {0: None, 1: None, 2: None}})}

		inp = dict_groups
		ref = dirdic
		pbar = tqdm()  # as function is always called from extract_features with percentage bar as parameter
		case = compute_direction(inp, pbar)

		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i], check_dtype=False)

	def test_compute_average_speed(self):
		avspeed = {
			312: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 312, 1: 312, 2: 312},
								  'x': {0: 405.29, 1: 405.31, 2: 405.31},
								  'y': {0: 417.76, 1: 417.37, 2: 417.07},
								  'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
								  'average_speed': {0: 0.195256, 1: 0.345256, 2: 0.150000},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
								  'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 511, 1: 511, 2: 511},
								  'x': {0: 369.99, 1: 370.01, 2: 370.01},
								  'y': {0: 428.78, 1: 428.82, 2: 428.85},
								  'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								  'average_speed': {0: 0.02236067977500298, 1: 0.03736067977501776, 2: 0.01500000000001478},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
								  'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 607, 1: 607, 2: 607},
								  'x': {0: 390.33, 1: 390.25, 2: 390.17},
								  'y': {0: 405.89, 1: 405.89, 2: 405.88},
								  'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								  'average_speed': {0: 0.040000, 1: 0.080311, 2: 0.040311},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
								  'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 811, 1: 811, 2: 811},
								  'x': {0: 445.15, 1: 445.48, 2: 445.77},
								  'y': {0: 411.94, 1: 412.26, 2: 412.61},
								  'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
								  'average_speed': {0: 0.229837, 1: 0.457103, 2: 0.227266},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
								  'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 905, 1: 905, 2: 905},
								  'x': {0: 366.06, 1: 365.86, 2: 365.7},
								  'y': {0: 451.76, 1: 451.76, 2: 451.76},
								  'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								  'average_speed': {0: 0.10, 1: 0.18, 2: 0.08},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: 180.0, 2: 180.0},
								  'stopped': {0: None, 1: None, 2: None}})}
		distdic = {
			312: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 312, 1: 312, 2: 312},
				'x': {0: 405.29, 1: 405.31, 2: 405.31},
				'y': {0: 417.76, 1: 417.37, 2: 417.07},
				'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
				'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 511, 1: 511, 2: 511},
				'x': {0: 369.99, 1: 370.01, 2: 370.01},
				'y': {0: 428.78, 1: 428.82, 2: 428.85},
				'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
				'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 607, 1: 607, 2: 607},
				'x': {0: 390.33, 1: 390.25, 2: 390.17},
				'y': {0: 405.89, 1: 405.89, 2: 405.88},
				'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
				'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 811, 1: 811, 2: 811},
				'x': {0: 445.15, 1: 445.48, 2: 445.77},
				'y': {0: 411.94, 1: 412.26, 2: 412.61},
				'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
				'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
				'time': {0: 1, 1: 2, 2: 3},
				'animal_id': {0: 905, 1: 905, 2: 905},
				'x': {0: 366.06, 1: 365.86, 2: 365.7},
				'y': {0: 451.76, 1: 451.76, 2: 451.76},
				'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
				'average_speed': {0: None, 1: None, 2: None},
				'average_acceleration': {0: None, 1: None, 2: None},
				'direction': {0: None, 1: 180.0, 2: 180.0},
				'stopped': {0: None, 1: None, 2: None}})}

		ref = avspeed
		inp = distdic
		case = compute_average_speed(inp, fps = 2)
		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i], check_dtype=False)

	def test_average_acceleration(self):  # test fails because we define average_acc[0] = average_acc[1] but in test as na.
		ref = avaccel
		inp = avspeed
		case = compute_average_acceleration(inp, fps = 2)
		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i], check_dtype=False)

	def test_computing_stops(self):
		ref = stops
		inp = avaccel
		case = computing_stops(inp, threshold_speed = 0.1)
		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i], check_dtype=False)

	def test_centoid_medoid_computation(self):
		inp = pd.DataFrame(records)
		ref = pd.DataFrame(medoids)

		case = centroid_medoid_computation(inp)
		case = case.sort_values(by=['time', 'animal_id']).reset_index(drop=True)
		pd.testing.assert_frame_equal(ref, case, check_dtype = False)



	def test_extract_features(self):
		"""
		Testing feature extraction for both, numeric and string time indices
		:return: Equality of records vs features dataset for both time-datatypes
		"""

		# Numeric datatype
		ref = pd.DataFrame(feats)

		# Adapt reference to sorted output
		ref.sort_values(['time', 'animal_id'], ascending=True, inplace=True)
		ref.reset_index(drop=True, inplace=True)
		inp = pd.DataFrame(records)
		case = extract_features(inp, fps = 1, stop_threshold=0.5).round(4)

		# String datatype
		ref_time = pd.DataFrame(timestamp_feats)

		# Adapt reference to sorted output
		ref_time.sort_values(['time', 'animal_id'], ascending=True, inplace=True)
		ref_time.reset_index(drop=True, inplace=True)
		inp_time = pd.DataFrame(records_timestring)
		inp_time["time"] = pd.to_datetime(inp_time["time"])
		case_time = extract_features(inp_time, fps='1s', stop_threshold=0.1).round(4)

		# Testing function on numeric "time" datatype
		pd.testing.assert_frame_equal(ref, case, check_dtype=False)

		# Testing function on string "time" datatype
		pd.testing.assert_frame_equal(ref_time, case_time, check_dtype=False)


	def test_ts_feature(self):
		ref = pd.DataFrame(ts_feat).round(4)
		inp = pd.DataFrame(feats)
		case = ts_feature(inp, "autocorrelation").round(4)
		ref = ref.rename_axis('variable', axis=1)
		ref = ref.rename_axis("id", axis=0)
		ref, case = ref, case.iloc[:,10:13]
		pd.testing.assert_frame_equal(ref,case, check_dtype=False)

	def test_all_ts_features(self):
		ref = pd.DataFrame(ts_feats).round(4)
		inp = pd.DataFrame(feats)
		case = ts_all_features(inp).round(4).iloc[:, 797:800]
		pd.testing.assert_frame_equal(ref, case, check_dtype=False)

	def test_outlier_detection(self):
		feats_knn = pd.DataFrame(
			{'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3, 12: 1, 13: 2, 14: 3},
			 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811,
						   11: 811,
						   12: 905, 13: 905, 14: 905},
			 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
				   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
			 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
				   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
			 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
						  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
			 'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3), 3: (0.0, 0.0), 4: (0.02, 0.04),
						   5: (0.0, 0.03), 6: (0.0, 0.0), 7: (-0.08, 0.0), 8: (-0.08, -0.01),
						   9: (0.0, 0.0), 10: (0.33, 0.32), 11: (0.29, 0.35), 12: (0.0, 0.0), 13: (-0.2, 0.0),
						   14: (-0.16, 0.0)},
			 'turning': {0: 0.000000, 1: 0.0, 2: 0.9987, 3: 0.000000, 4: 0.0, 5: 0.8944, 6: 0.000000, 7: 0.000000,
						 8: 0.9923, 9: 0.000000,
						 10: 0.0, 11: 0.9941, 12: 0.000000, 13: 0.000000, 14: 1.000000},
			 'average_speed': {0: 0.1953, 1: 0.3453, 2: 0.1500, 3: 0.0224, 4: 0.0374, 5: 0.0150, 6: 0.0400, 7: 0.0803,
							   8: 0.0403,
							   9: 0.2298, 10: 0.4571, 11: 0.2273, 12: 0.10, 13: 0.18, 14: 0.08},
			 'average_acceleration': {0: 0.0750, 1: -0.0226, 2: -0.0976, 3: 0.0075, 4: -0.0037, 5: -0.0112, 6: 0.0202,
									  7: 0.0002,
									  8: -0.0200, 9: 0.1136, 10: -0.0013, 11: -0.1149, 12: 0.04, 13: -0.01, 14: -0.05},
			 'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
						 14: 1}})

		feats_ecod = pd.DataFrame(
			{'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3, 12: 1, 13: 2, 14: 3},
			 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811,
						   11: 811,
						   12: 905, 13: 905, 14: 905},
			 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
				   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
			 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
				   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
			 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
						  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
			 'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3), 3: (0.0, 0.0), 4: (0.02, 0.04),
						   5: (0.0, 0.03), 6: (0.0, 0.0), 7: (-0.08, 0.0), 8: (-0.08, -0.01),
						   9: (0.0, 0.0), 10: (0.33, 0.32), 11: (0.29, 0.35), 12: (0.0, 0.0), 13: (-0.2, 0.0),
						   14: (-0.16, 0.0)},
			 'turning': {0: 0.000000, 1: 0.0, 2: 0.9987, 3: 0.000000, 4: 0.0, 5: 0.8944, 6: 0.000000, 7: 0.000000,
						 8: 0.9923, 9: 0.000000,
						 10: 0.0, 11: 0.9941, 12: 0.000000, 13: 0.000000, 14: 1.000000},
			 'average_speed': {0: 0.1953, 1: 0.3453, 2: 0.1500, 3: 0.0224, 4: 0.0374, 5: 0.0150, 6: 0.0400, 7: 0.0803,
							   8: 0.0403,
							   9: 0.2298, 10: 0.4571, 11: 0.2273, 12: 0.10, 13: 0.18, 14: 0.08},
			 'average_acceleration': {0: 0.0750, 1: -0.0226, 2: -0.0976, 3: 0.0075, 4: -0.0037, 5: -0.0112, 6: 0.0202,
									  7: 0.0002,
									  8: -0.0200, 9: 0.1136, 10: -0.0013, 11: -0.1149, 12: 0.04, 13: -0.01, 14: -0.05},
			 'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
						 14: 1}})

		ref_knn = pd.DataFrame(
			{'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3, 12: 1, 13: 2, 14: 3},
			 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811,
						   11: 811,
						   12: 905, 13: 905, 14: 905},
			 'outlier':{0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:1,11:0,12:0,13:0,14:0},
			 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
				   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
			 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
				   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
			 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
						  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
			 'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3), 3: (0.0, 0.0), 4: (0.02, 0.04),
						   5: (0.0, 0.03), 6: (0.0, 0.0), 7: (-0.08, 0.0), 8: (-0.08, -0.01),
						   9: (0.0, 0.0), 10: (0.33, 0.32), 11: (0.29, 0.35), 12: (0.0, 0.0), 13: (-0.2, 0.0),
						   14: (-0.16, 0.0)},
			 'turning': {0: 0.000000, 1: 0.0, 2: 0.9987, 3: 0.000000, 4: 0.0, 5: 0.8944, 6: 0.000000, 7: 0.000000,
						 8: 0.9923, 9: 0.000000,
						 10: 0.0, 11: 0.9941, 12: 0.000000, 13: 0.000000, 14: 1.000000},
			 'average_speed': {0: 0.1953, 1: 0.3453, 2: 0.1500, 3: 0.0224, 4: 0.0374, 5: 0.0150, 6: 0.0400, 7: 0.0803,
							   8: 0.0403,
							   9: 0.2298, 10: 0.4571, 11: 0.2273, 12: 0.10, 13: 0.18, 14: 0.08},
			 'average_acceleration': {0: 0.0750, 1: -0.0226, 2: -0.0976, 3: 0.0075, 4: -0.0037, 5: -0.0112, 6: 0.0202,
									  7: 0.0002,
									  8: -0.0200, 9: 0.1136, 10: -0.0013, 11: -0.1149, 12: 0.04, 13: -0.01, 14: -0.05},
			 'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
						 14: 1}})

		ref_ecod = pd.DataFrame(
			{'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3, 12: 1, 13: 2, 14: 3},
			 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811,
						   11: 811,
						   12: 905, 13: 905, 14: 905},
			 'outlier': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 0, 13: 0, 14: 0},
			 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
				   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
			 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
				   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
			 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
						  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
			 'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3), 3: (0.0, 0.0), 4: (0.02, 0.04),
						   5: (0.0, 0.03), 6: (0.0, 0.0), 7: (-0.08, 0.0), 8: (-0.08, -0.01),
						   9: (0.0, 0.0), 10: (0.33, 0.32), 11: (0.29, 0.35), 12: (0.0, 0.0), 13: (-0.2, 0.0),
						   14: (-0.16, 0.0)},
			 'turning': {0: 0.000000, 1: 0.0, 2: 0.9987, 3: 0.000000, 4: 0.0, 5: 0.8944, 6: 0.000000, 7: 0.000000,
						 8: 0.9923, 9: 0.000000,
						 10: 0.0, 11: 0.9941, 12: 0.000000, 13: 0.000000, 14: 1.000000},
			 'average_speed': {0: 0.1953, 1: 0.3453, 2: 0.1500, 3: 0.0224, 4: 0.0374, 5: 0.0150, 6: 0.0400, 7: 0.0803,
							   8: 0.0403,
							   9: 0.2298, 10: 0.4571, 11: 0.2273, 12: 0.10, 13: 0.18, 14: 0.08},
			 'average_acceleration': {0: 0.0750, 1: -0.0226, 2: -0.0976, 3: 0.0075, 4: -0.0037, 5: -0.0112, 6: 0.0202,
									  7: 0.0002,
									  8: -0.0200, 9: 0.1136, 10: -0.0013, 11: -0.1149, 12: 0.04, 13: -0.01, 14: -0.05},
			 'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
						 14: 1}})

		case_knn = outlier_detection(feats_knn, algorithm="KNN", contamination=0.2, n_neighbors=4)
		case_ecod = outlier_detection(feats_ecod, algorithm="ECOD")

		pd.testing.assert_frame_equal(ref_ecod, case_ecod, check_dtype=False)
		pd.testing.assert_frame_equal(ref_knn, case_knn, check_dtype=False)



	def test_split_movement_trajectory(self):
		inp = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
										  9: 905, 10: 312, 11: 511},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31, 11: 370.01},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07, 11: 428.85},
							'distance': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447,
										 7: 0.08, 8: 0.4597, 9: 0.2, 10: 0.3, 11: 0.03},
							'average_speed': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447,
											  7: 0.08, 8: 0.4597, 9: 0.2, 10: 0.3, 11: 0.03},
							'average_acceleration': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447,
													 7: 0.08, 8: 0.4597, 9: 0.2, 10: -0.0905, 11: -0.0147},
							'direction': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -87.0643, 6: 63.4349,
										  7: 180.0, 8: 44.1186, 9: 180.0, 10: -90.0, 11: 90.0},
							'turning': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.000000, 6: 0.000000,
										7: 0.000000, 8: 0.000000, 9: 0.000000, 10: 0.998688, 11: 0.894427},
							'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1}})
		ref = {
			312: [
				pd.DataFrame({'time': {0: 1},
							  'animal_id': {0: 312},
							  'x': {0: 405.29},
							  'y': {0: 417.76},
							  'distance': {0: 0.0},
							  'average_speed': {0: 0.0},
							  'average_acceleration': {0: 0.0},
							  'direction': {0: 0.0},
							  'turning': {0: 0.0},
							  'stopped': {0: 1}}),
				pd.DataFrame({'time': {5: 2, 10: 3},
							  'animal_id': {5: 312, 10: 312},
							  'x': {5: 405.31, 10: 405.31},
							  'y': {5: 417.37, 10: 417.07},
							  'distance': {5: 0.3905, 10: 0.3},
							  'average_speed': {5: 0.3905, 10: 0.3},
							  'average_acceleration': {5: 0.3905, 10: -0.0905},
							  'direction': {5: -87.0643, 10: -90.0},
							  'turning': {5: 0.000000, 10: 0.998688},
							  'stopped': {5: 0, 10: 0}})
			],
			511:[
				pd.DataFrame({'time': {1: 1, 6: 2, 11: 3},
							  'animal_id': {1: 511, 6: 511, 11: 511},
							  'x': {1: 369.99, 6: 370.01, 11: 370.01},
							  'y': {1: 428.78, 6: 428.82, 11: 428.85},
							  'distance': {1: 0.0, 6: 0.0447, 11: 0.03},
							  'average_speed': {1: 0.0, 6: 0.0447, 11: 0.03},
							  'average_acceleration': {1: 0.0, 6: 0.0447, 11: -0.0147},
							  'direction': {1: 0.0, 6: 63.4349, 11: 90.0},
							  'turning': {1: 0.0, 6: 0.000000, 11: 0.894427},
							  'stopped': {1: 1, 6: 1, 11: 1}})
			],
			607:[
				pd.DataFrame({'time': {2: 1, 7: 2},
							  'animal_id': {2: 607, 7: 607},
							  'x': {2: 390.33, 7: 390.25},
							  'y': {2: 405.89,7: 405.89},
							  'distance': {2: 0.0, 7: 0.08},
							  'average_speed': {2: 0.0, 7: 0.08},
							  'average_acceleration': {2: 0.0, 7: 0.08},
							  'direction': {2: 0.0, 7: 180.0},
							  'turning': {2: 0.0, 7: 0.000000},
							  'stopped': {2: 1, 7: 1}})
			],
			811:[
				pd.DataFrame({'time': {3: 1},
							  'animal_id': {3: 811},
							  'x': {3: 445.15},
							  'y': {3: 411.94},
							  'distance': {3: 0.0},
							  'average_speed': {3: 0.0},
							  'average_acceleration': {3: 0.0},
							  'direction': {3: 0.0},
							  'turning': {3: 0.0},
							  'stopped': {3: 1}}),
				pd.DataFrame({'time': {8: 2},
							  'animal_id': {8: 811},
							  'x': {8: 445.48},
							  'y': {8: 412.26},
							  'distance': {8: 0.4597},
							  'average_speed': {8: 0.4597},
							  'average_acceleration': {8: 0.4597},
							  'direction': {8: 44.1186},
							  'turning': {8: 0.000000},
							  'stopped': {8: 0}})
			],
			905:[
				pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3},
							  'animal_id': {4: 905,	9: 905},
							  'x': {4: 366.06, 9: 365.86},
							  'y': {4: 451.76, 9: 451.76},
							  'distance': {4: 0.0, 9: 0.2},
							  'average_speed': {4: 0.0, 9: 0.2},
							  'average_acceleration': {4: 0.0, 9: 0.2},
							  'direction': {4: 0.0, 9: 180.0},
							  'turning': {4: 0.0, 9: 0.000000},
							  'stopped': {4: 1, 9: 1}})
			]
		}

		case = split_movement_trajectory(inp, stop_threshold=0.25)
		pd.testing.assert_frame_equal(ref[312][0].reset_index(drop=True), case[312][0], check_dtype=False)
		pd.testing.assert_frame_equal(ref[312][1].reset_index(drop=True), case[312][1], check_dtype=False)
		pd.testing.assert_frame_equal(ref[511][0].reset_index(drop=True), case[511][0], check_dtype=False)
		pd.testing.assert_frame_equal(ref[811][0].reset_index(drop=True), case[811][0], check_dtype=False)
		pd.testing.assert_frame_equal(ref[811][1].reset_index(drop=True), case[811][1], check_dtype=False)


	def test_movement_stopping_durations(self):
		inp = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
										  9: 905, 10: 312, 11: 511},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31, 11: 370.01},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07, 11: 428.85},
							'distance': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447,
										 7: 0.08, 8: 0.4597, 9: 0.2, 10: 0.3, 11: 0.03},
							'average_speed': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447,
											  7: 0.08, 8: 0.4597, 9: 0.2, 10: 0.3, 11: 0.03},
							'average_acceleration': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447,
													 7: 0.08, 8: 0.4597, 9: 0.2, 10: -0.0905, 11: -0.0147},
							'direction': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -87.0643, 6: 63.4349,
										  7: 180.0, 8: 44.1186, 9: 180.0, 10: -90.0, 11: 90.0},
							'turning': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.000000, 6: 0.000000,
										7: 0.000000, 8: 0.000000, 9: 0.000000, 10: 0.998688, 11: 0.894427},
							'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1}})
		ref_312 = pd.DataFrame({'animal_id': 312, 'Duration of phase 1 (stopping)': 1, 'Duration of phase 2 (moving)': 2}, index=[0])
		ref_905 = pd.DataFrame({'animal_id': 905, 'Duration of phase 1 (stopping)': 2}, index=[0])
		case = movement_stopping_durations(inp, stop_threshold=0.25)
		pd.testing.assert_frame_equal(ref_312, case[312], check_dtype=False)
		pd.testing.assert_frame_equal(ref_905, case[905], check_dtype=False)

	def test_hausdorrf_distance(self):
		inp = pd.DataFrame(euclidean)
		case = hausdorff_distance(inp, 312, 511)
		self.assertAlmostEqual(case, 37.20420003171686, places=7)

	def test_outlier_by_threshold(self):
		feats = pd.DataFrame(
			{'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3, 12: 1, 13: 2, 14: 3},
			 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811,
						   11: 811,
						   12: 905, 13: 905, 14: 905},
			 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
				   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
			 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
				   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
			 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
						  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
			 'direction': {0: (0.0, 0.0), 1: (0.02, -0.39), 2: (0.0, -0.3), 3: (0.0, 0.0), 4: (0.02, 0.04),
						   5: (0.0, 0.03), 6: (0.0, 0.0), 7: (-0.08, 0.0), 8: (-0.08, -0.01),
						   9: (0.0, 0.0), 10: (0.33, 0.32), 11: (0.29, 0.35), 12: (0.0, 0.0), 13: (-0.2, 0.0),
						   14: (-0.16, 0.0)},
			 'turning': {0: 0.000000, 1: 0.0, 2: 0.9987, 3: 0.000000, 4: 0.0, 5: 0.8944, 6: 0.000000, 7: 0.000000,
						 8: 0.9923, 9: 0.000000,
						 10: 0.0, 11: 0.9941, 12: 0.000000, 13: 0.000000, 14: 1.000000},
			 'average_speed': {0: 0.1953, 1: 0.3453, 2: 0.1500, 3: 0.0224, 4: 0.0374, 5: 0.0150, 6: 0.0400, 7: 0.0803,
							   8: 0.0403,
							   9: 0.2298, 10: 0.4571, 11: 0.2273, 12: 0.10, 13: 0.18, 14: 0.08},
			 'average_acceleration': {0: 0.0750, 1: -0.0226, 2: -0.0976, 3: 0.0075, 4: -0.0037, 5: -0.0112, 6: 0.0202,
									  7: 0.0002,
									  8: -0.0200, 9: 0.1136, 10: -0.0013, 11: -0.1149, 12: 0.04, 13: -0.01, 14: -0.05},
			 'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1,
						 14: 1}})

		ref = pd.DataFrame(
			{'time': {0: 1, 2: 3, 12: 1, 13: 2},
			 'animal_id': {0: 312, 2: 312, 12: 905, 13: 905},
			 'x': {0: 405.29, 2: 405.31, 12: 366.06, 13: 365.86},
			 'y': {0: 417.76, 2: 417.07, 12: 451.76, 13: 451.76},
			 'distance': {0: 0.0, 2: 0.3, 12: 0.0, 13: 0.2},
			 'direction': {0: (0.0, 0.0), 2: (0.0, -0.3), 12: (0.0, 0.0), 13: (-0.2, 0.0)},
			 'turning': {0: 0.000000, 2: 0.9987, 12: 0.000000, 13: 0.000000},
			 'average_speed': {0: 0.1953, 2: 0.1500, 12: 0.10, 13: 0.18},
			 'average_acceleration': {0: 0.0750, 2: -0.0976, 12: 0.04, 13: -0.01},
			 'stopped': {0: 1, 2: 1, 12: 1, 13: 1},
			 'outlier_by_threshold': {0: 0, 2: 0, 12: 0, 13: 0}})

		case = outlier_by_threshold(feats, feature_thresholds={'average_speed': [0.10, 0.20]}, remove=True)
		pd.testing.assert_frame_equal(ref, case, check_dtype=False)


	def test_segment_data(self):
		inp = pd.DataFrame(records)
		ref = {
			312: [
				pd.DataFrame({'time': {0: 1, 1:2},
							  'animal_id': {0: 312, 1:312},
							  'x': {0: 405.29, 1:405.31},
							  'y': {0: 417.76, 1:417.37},
							  'distance': {0: 0.0, 1:0.39051},
							  'direction': {0: (0.0,0.0),1: (0.02,-0.39)},
							  'turning': {0: 0.0, 1: 0.0},
							  'average_speed': {0: 0.19525624189765908, 1: 0.3452562418976648},
							  'average_acceleration': {0: 0.07500000000000287, 1:-0.022628120948826685},
							  'stopped': {0: 1, 1:1}}),
				pd.DataFrame({'time': {0: 3},
							  'animal_id': {0: 312},
							  'x': {0: 405.31},
							  'y': {0: 417.07},
							  'distance': {0: 0.3},
							  'direction': {0: (0.0,-0.3)},
							  'turning': {0: 0.99869},
							  'average_speed': {0: 0.15},
							  'average_acceleration': {0: -0.09762812094882953},
							  'stopped': {0: 1}})
			]
		}
		case = segment_data(inp, feature='average_speed', threshold=0.18, fps = 2, stop_threshold=0.4)
		pd.testing.assert_frame_equal(ref[312][0], case[312][0], check_dtype=False)
		pd.testing.assert_frame_equal(ref[312][1], case[312][1], check_dtype=False)




if __name__ == '__main__':
    unittest.main()
