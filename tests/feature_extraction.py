import os
import unittest
from pandas import Timestamp
import pandas as pd
import numpy as np
import math
import pickle
from pandas.testing import assert_frame_equal


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
			 "312": {0: 0.0, 1: 0.502, 2: 0.3191, 3: 0.5137, 4: 0.8884, 5: 0.0, 6: 0.5066, 7: 0.313, 8: 0.5139,
					 9: 0.897, 10: 0.0, 11: 0.5102, 12: 0.3086, 13: 0.5146, 14: 0.9036},
			 "511": {0: 0.502, 1: 0.0, 2: 0.5599, 3: 1.0079, 4: 0.5033, 5: 0.5066, 6: 0.0, 7: 0.5601, 8: 1.0093,
					 9: 0.5027, 10: 0.5102, 11: 0.0, 12: 0.5604, 13: 1.0102, 14: 0.5022},
			 "607": {0: 0.3191, 1: 0.5599, 2: 0.0, 3: 0.6972, 4: 1.0447, 5: 0.313, 6: 0.5601, 7: 0.0, 8: 0.7036,
					 9: 1.0452, 10: 0.3086, 11: 0.5604, 12: 0.0, 13: 0.7097, 14: 1.0457},
			 "811": {0: 0.5137, 1: 1.0079, 2: 0.6972, 3: 0.0, 4: 1.3149, 5: 0.5139, 6: 1.0093, 7: 0.7036,
					 8: 0.0, 9: 1.3153, 10: 0.5146, 11: 1.0102, 12: 0.7097, 13: 0.0, 14: 1.3146},
			 "905": {0: 0.8884, 1: 0.5033, 2: 1.0447, 3: 1.3149, 4: 0.0, 5: 0.897, 6: 0.5027, 7: 1.0452,
					 8: 1.3153, 9: 0.0, 10: 0.9036, 11: 0.5022, 12: 1.0457, 13: 1.3146, 14: 0.0}}

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
							'average_speed': {0: 0.0, 1: 0.39051248379531817, 2: 0.3000000000000114},
							'average_acceleration': {0: None, 1: None, 2: None},
							'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
							'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 511, 1: 511, 2: 511},
							   'x': {0: 369.99, 1: 370.01, 2: 370.01}, 'y': {0: 428.78, 1: 428.82, 2: 428.85},
							   'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
							   'average_speed': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
							   'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 607, 1: 607, 2: 607},
							   'x': {0: 390.33, 1: 390.25, 2: 390.17}, 'y': {0: 405.89, 1: 405.89, 2: 405.88},
							   'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
							   'average_speed': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
							   'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 811, 1: 811, 2: 811},
							   'x': {0: 445.15, 1: 445.48, 2: 445.77}, 'y': {0: 411.94, 1: 412.26, 2: 412.61},
							   'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
							   'average_speed': {0: 0.0, 1: 0.45967379738247277, 2: 0.4545327270945347},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
							   'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 905, 1: 905, 2: 905},
							   'x': {0: 366.06, 1: 365.86, 2: 365.7}, 'y': {0: 451.76, 1: 451.76, 2: 451.76},
							   'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
							   'average_speed': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
							   'average_acceleration': {0: None, 1: None, 2: None},
							   'direction': {0: None, 1: 180.0, 2: 180.0}, 'stopped': {0: None, 1: None, 2: None}})}

avaccel = {
			312 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 312, 1: 312, 2: 312},
					   			'x': {0: 405.29, 1: 405.31, 2: 405.31},
								'y': {0: 417.76, 1: 417.37, 2: 417.07},
								'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
								'average_speed': {0: 0.0, 1: 0.39051248379531817, 2: 0.3000000000000114},
								'average_acceleration': {0: None, 1: 0.39051248379531817, 2: -0.09051248379530674},
								'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
								'stopped': {0: None, 1: None, 2: None}} ),
			511 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
								 'animal_id': {0: 511, 1: 511, 2: 511}, 'x': {0: 369.99,
																			  1: 370.01, 2: 370.01},
								 'y': {0: 428.78, 1: 428.82, 2: 428.85},
								 'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								 'average_speed': {0: 0.0,1: 0.04472135955000596, 2: 0.03000000000002956},
								 'average_acceleration': {0: None, 1: 0.04472135955000596, 2: -0.014721359549976401},
								 'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
								 'stopped': {0: None, 1: None, 2: None}} ),
			607 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 607, 1: 607, 2: 607},
								'x': {0: 390.33, 1: 390.25, 2: 390.17},
								'y': {0: 405.89, 1: 405.89, 2: 405.88},
								'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								'average_speed': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								'average_acceleration': {0: None, 1: 0.07999999999998408, 2: 0.0006225774829844988},
								'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
								'stopped': {0: None, 1: None, 2: None}} ),
			811 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 811, 1: 811, 2: 811},
								'x': {0: 445.15, 1: 445.48, 2: 445.77},
								'y': {0: 411.94, 1: 412.26, 2: 412.61},
								'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
								'average_speed': {0: 0.0, 1: 0.45967379738247277, 2: 0.4545327270945347},
								'average_acceleration': {0: None, 1: 0.45967379738247277, 2: -0.005141070287938088},
								'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
								'stopped': {0: None, 1: None, 2: None}} ),
			905 : pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 905, 1: 905, 2: 905},
								'x': {0: 366.06, 1: 365.86, 2: 365.7},
								'y': {0: 451.76, 1: 451.76, 2: 451.76},
								'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								'average_speed': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								'average_acceleration': {0: None, 1: 0.19999999999998863, 2: -0.03999999999996362},
								'direction': {0: None, 1: 180.0, 2: 180.0},
								'stopped': {0: None, 1: None, 2: None}})}

stops = {
			312 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
								'animal_id': {0: 312, 1: 312, 2: 312},
								'x': {0: 405.29, 1: 405.31,	2: 405.31}, 'y': {0: 417.76, 1: 417.37, 2: 417.07},
								  'distance': {0: 0.0, 1: 0.39051248379531817, 2: 0.30000000000001137},
								  'average_speed': {0: 0.0, 1: 0.39051248379531817, 2: 0.3000000000000114},
								  'average_acceleration': {0: None, 1: 0.39051248379531817, 2: -0.09051248379530674},
								  'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
								  'stopped': {0: 1, 1: 1, 2: 1}} ),
			511 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 511, 1: 511, 2: 511},
								 'x': {0: 369.99, 1: 370.01, 2: 370.01}, 'y': {0: 428.78, 1: 428.82, 2: 428.85},
								 'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								 'average_speed': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								 'average_acceleration': {0: None, 1: 0.04472135955000596, 2: -0.014721359549976401},
								 'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
								 'stopped': {0: 1, 1: 1, 2: 1}} ),
			607 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 607, 1: 607, 2: 607},
								 'x': {0: 390.33, 1: 390.25, 2: 390.17}, 'y': {0: 405.89, 1: 405.89, 2: 405.88},
								 'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								 'average_speed': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								 'average_acceleration': {0: None, 1: 0.07999999999998408, 2: 0.0006225774829844988},
								 'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
								 'stopped': {0: 1, 1: 1, 2: 1}} ),
			811 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 811, 1: 811, 2: 811},
								 'x': {0: 445.15, 1: 445.48, 2: 445.77}, 'y': {0: 411.94, 1: 412.26, 2: 412.61},
								 'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
								 'average_speed': {0: 0.0, 1: 0.45967379738247277, 2: 0.4545327270945347},
								 'average_acceleration': {0: None, 1: 0.45967379738247277, 2: -0.005141070287938088},
								 'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
								 'stopped': {0: 1, 1: 1, 2: 1}} ),
			905 : pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3}, 'animal_id': {0: 905, 1: 905, 2: 905},
								 'x': {0: 366.06, 1: 365.86, 2: 365.7}, 'y': {0: 451.76, 1: 451.76, 2: 451.76},
								 'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								 'average_speed': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								 'average_acceleration': {0: None, 1: 0.19999999999998863, 2: -0.03999999999996362},
								 'direction': {0: None, 1: 180.0, 2: 180.0}, 'stopped': {0: 1, 1: 1, 2: 1}} )}

feats = {'time': {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: 2, 8: 3, 9: 1, 10: 2, 11: 3, 12: 1, 13: 2, 14: 3},
		 'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607, 8: 607, 9: 811, 10: 811, 11: 811,
					   12: 905, 13: 905, 14: 905},
		 'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33, 7: 390.25, 8: 390.17,
			   9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06, 13: 365.86, 14: 365.7},
		 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78, 4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89, 8: 405.88,
			   9: 411.94, 10: 412.26, 11: 412.61, 12: 451.76, 13: 451.76, 14: 451.76},
		 'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806, 9: 0.0,
					  10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
		 'average_speed': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0, 7: 0.08, 8: 0.0806,
						   9: 0.0, 10: 0.4597, 11: 0.4545, 12: 0.0, 13: 0.2, 14: 0.16},
		 'average_acceleration': {0: 0.0, 1: 0.3905, 2: -0.0905, 3: 0.0, 4: 0.0447, 5: -0.0147, 6: 0.0, 7: 0.08,
								  8: 0.0006, 9: 0.0, 10: 0.4597, 11: -0.0051, 12: 0.0, 13: 0.2, 14: -0.04},
		 'direction': {0: 0.0, 1: -87.0643, 2: -90.0, 3: 0.0, 4: 63.4349, 5: 90.0, 6: 0.0, 7: 180.0, 8: -172.875,
					   9: 0.0, 10: 44.1186, 11: 50.3558, 12: 0.0, 13: 180.0, 14: 180.0},
		 'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1}}

timestamp_feats = {'time': {0: Timestamp('2020-03-27 11:57:07'), 1: Timestamp('2020-03-27 11:57:09'),
							2: Timestamp('2020-03-27 11:57:11'), 3: Timestamp('2020-03-27 11:57:07'),
							4: Timestamp('2020-03-27 11:57:09'), 5: Timestamp('2020-03-27 11:57:11'),
							6: Timestamp('2020-03-27 11:57:07'), 7: Timestamp('2020-03-27 11:57:09'),
							8: Timestamp('2020-03-27 11:57:11'), 9: Timestamp('2020-03-27 11:57:07'),
							10: Timestamp('2020-03-27 11:57:09'), 11: Timestamp('2020-03-27 11:57:11'),
							12: Timestamp('2020-03-27 11:57:07'), 13: Timestamp('2020-03-27 11:57:09'),
							14: Timestamp('2020-03-27 11:57:11')},
				   'animal_id': {0: 312, 1: 312, 2: 312, 3: 511, 4: 511, 5: 511, 6: 607, 7: 607,
								 8: 607, 9: 811, 10: 811, 11: 811, 12: 905, 13: 905, 14: 905},
				   'x': {0: 405.29, 1: 405.31, 2: 405.31, 3: 369.99, 4: 370.01, 5: 370.01, 6: 390.33,
						 7: 390.25, 8: 390.17, 9: 445.15, 10: 445.48, 11: 445.77, 12: 366.06,
						 13: 365.86, 14: 365.7}, 'y': {0: 417.76, 1: 417.37, 2: 417.07, 3: 428.78,
													   4: 428.82, 5: 428.85, 6: 405.89, 7: 405.89,
													   8: 405.88, 9: 411.94, 10: 412.26, 11: 412.61,
													   12: 451.76, 13: 451.76, 14: 451.76},
				   'distance': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0,
								7: 0.08, 8: 0.0806, 9: 0.0, 10: 0.4597, 11: 0.4545, 12: 0.0,
								13: 0.2, 14: 0.16},
				   'average_speed': {0: 0.0, 1: 0.3905, 2: 0.3, 3: 0.0, 4: 0.0447, 5: 0.03, 6: 0.0,
									 7: 0.08, 8: 0.0806, 9: 0.0, 10: 0.4597, 11: 0.4545, 12: 0.0,
									 13: 0.2, 14: 0.16},
				   'average_acceleration': {0: 0.0, 1: 0.3905, 2: -0.0905, 3: 0.0, 4: 0.0447,
											5: -0.0147, 6: 0.0, 7: 0.08, 8: 0.0006, 9: 0.0,
											10: 0.4597, 11: -0.0051, 12: 0.0, 13: 0.2, 14: -0.04},
				   'direction': {0: 0.0, 1: -87.0643, 2: -90.0, 3: 0.0, 4: 63.4349, 5: 90.0,
								 6: 0.0, 7: 180.0, 8: -172.875, 9: 0.0, 10: 44.1186, 11: 50.3558,
								 12: 0.0, 13: 180.0, 14: 180.0},
				   'stopped': {0: 1, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
							   10: 0, 11: 0, 12: 1, 13: 0, 14: 0}}

ts_feat = {'average_acceleration__autocorrelation__lag_0': {312: 1.0, 511: 1.0, 607: 1.0, 811: 1.0, 905: 1.0},
		   'average_acceleration__autocorrelation__lag_1': {312: -0.9687, 511: -0.9436, 607: -1.0, 811: -0.9999,
															905: -0.9758},
		   'average_acceleration__autocorrelation__lag_2': {312: 0.4373, 511: 0.3871, 607: 0.4999, 811: 0.4998,
															905: 0.4516},
		   'average_acceleration__autocorrelation__lag_3': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_acceleration__autocorrelation__lag_4': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_acceleration__autocorrelation__lag_5': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_acceleration__autocorrelation__lag_6': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_acceleration__autocorrelation__lag_7': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_acceleration__autocorrelation__lag_8': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_acceleration__autocorrelation__lag_9': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_speed__autocorrelation__lag_0': {312: 1.0, 511: 1.0, 607: 1.0, 811: 1.0, 905: 1.0},
		   'average_speed__autocorrelation__lag_1': {312: -0.4615, 511: -0.5665, 607: -0.2444, 811: -0.2586,
													 905: -0.4286},
		   'average_speed__autocorrelation__lag_2': {312: -0.5771, 511: -0.367, 607: -1.0112, 811: -0.9828,
													 905: -0.6429},
		   'average_speed__autocorrelation__lag_3': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_speed__autocorrelation__lag_4': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_speed__autocorrelation__lag_5': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_speed__autocorrelation__lag_6': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_speed__autocorrelation__lag_7': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_speed__autocorrelation__lag_8': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'average_speed__autocorrelation__lag_9': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'direction__autocorrelation__lag_0': {312: 1.0, 511: 1.0, 607: 1.0, 811: 1.0, 905: 1.0},
		   'direction__autocorrelation__lag_1': {312: -0.2256, 511: -0.053, 607: -0.76, 811: -0.1587, 905: -0.25},
		   'direction__autocorrelation__lag_2': {312: -1.0489, 511: -1.394, 607: 0.0201, 811: -1.1826, 905: -1.0},
		   'direction__autocorrelation__lag_3': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'direction__autocorrelation__lag_4': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'direction__autocorrelation__lag_5': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'direction__autocorrelation__lag_6': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'direction__autocorrelation__lag_7': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'direction__autocorrelation__lag_8': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'direction__autocorrelation__lag_9': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'distance__autocorrelation__lag_0': {312: 1.0, 511: 1.0, 607: 1.0, 811: 1.0, 905: 1.0},
		   'distance__autocorrelation__lag_1': {312: -0.4615, 511: -0.5665, 607: -0.2444, 811: -0.2586, 905: -0.4286},
		   'distance__autocorrelation__lag_2': {312: -0.5771, 511: -0.367, 607: -1.0112, 811: -0.9828, 905: -0.6429},
		   'distance__autocorrelation__lag_3': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'distance__autocorrelation__lag_4': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'distance__autocorrelation__lag_5': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'distance__autocorrelation__lag_6': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'distance__autocorrelation__lag_7': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'distance__autocorrelation__lag_8': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'distance__autocorrelation__lag_9': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'x__autocorrelation__lag_0': {312: 1.0, 511: 1.0, 607: 1.0, 811: 1.0, 905: 1.0},
		   'x__autocorrelation__lag_1': {312: -0.25, 511: -0.25, 607: 0.0, 811: -0.0014, 905: -0.0041},
		   'x__autocorrelation__lag_2': {312: -1.0, 511: -1.0, 607: -1.5, 811: -1.4972, 905: -1.4918},
		   'x__autocorrelation__lag_3': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'x__autocorrelation__lag_4': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'x__autocorrelation__lag_5': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'x__autocorrelation__lag_6': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'x__autocorrelation__lag_7': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'x__autocorrelation__lag_8': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'x__autocorrelation__lag_9': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'y__autocorrelation__lag_0': {312: 1.0, 511: 1.0, 607: 1.0, 811: 1.0, 905: None},
		   'y__autocorrelation__lag_1': {312: -0.0056, 511: -0.0068, 607: -0.25, 811: -0.0007, 905: None},
		   'y__autocorrelation__lag_2': {312: -1.4887, 511: -1.4865, 607: -1.0, 811: -1.4987, 905: None},
		   'y__autocorrelation__lag_3': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'y__autocorrelation__lag_4': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'y__autocorrelation__lag_5': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'y__autocorrelation__lag_6': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'y__autocorrelation__lag_7': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'y__autocorrelation__lag_8': {312: None, 511: None, 607: None, 811: None, 905: None},
		   'y__autocorrelation__lag_9': {312: None, 511: None, 607: None, 811: None, 905: None}}

ts_feats = {'average_acceleration__agg_linear_trend__f_agg_"max"__chunk_len_5__attr_"slope"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"max"__chunk_len_5__attr_"stderr"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"intercept"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"rvalue"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"slope"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_10__attr_"stderr"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"intercept"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"rvalue"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"slope"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_50__attr_"stderr"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"intercept"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"rvalue"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"slope"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"mean"__chunk_len_5__attr_"stderr"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0},
			'average_acceleration__agg_linear_trend__f_agg_"min"__chunk_len_10__attr_"intercept"':
				{312: 0.0, 511: 0.0, 607: 0.0, 811: 0.0, 905: 0.0}}


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

	def test_eucledian_dist(self):
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


	def test_compute_distance_and_direction(self):
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
		case = compute_distance_and_direction(inp)

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
								  'average_speed': {0: 0.0, 1: 0.39051248379531817, 2: 0.3000000000000114},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: -87.0643265535814, 2: -90.0},
								  'stopped': {0: None, 1: None, 2: None}}),
			511: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 511, 1: 511, 2: 511},
								  'x': {0: 369.99, 1: 370.01, 2: 370.01},
								  'y': {0: 428.78, 1: 428.82, 2: 428.85},
								  'distance': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								  'average_speed': {0: 0.0, 1: 0.04472135955000596, 2: 0.03000000000002956},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: 63.434948822954574, 2: 90.0},
								  'stopped': {0: None, 1: None, 2: None}}),
			607: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 607, 1: 607, 2: 607},
								  'x': {0: 390.33, 1: 390.25, 2: 390.17},
								  'y': {0: 405.89, 1: 405.89, 2: 405.88},
								  'distance': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								  'average_speed': {0: 0.0, 1: 0.07999999999998408, 2: 0.08062257748296858},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: 180.0, 2: -172.87498365110324},
								  'stopped': {0: None, 1: None, 2: None}}),
			811: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 811, 1: 811, 2: 811},
								  'x': {0: 445.15, 1: 445.48, 2: 445.77},
								  'y': {0: 411.94, 1: 412.26, 2: 412.61},
								  'distance': {0: 0.0, 1: 0.45967379738247277, 2: 0.45453272709453474},
								  'average_speed': {0: 0.0, 1: 0.45967379738247277, 2: 0.4545327270945347},
								  'average_acceleration': {0: None, 1: None, 2: None},
								  'direction': {0: None, 1: 44.1185960034137, 2: 50.35582504286055},
								  'stopped': {0: None, 1: None, 2: None}}),
			905: pd.DataFrame({
								  'time': {0: 1, 1: 2, 2: 3},
								  'animal_id': {0: 905, 1: 905, 2: 905},
								  'x': {0: 366.06, 1: 365.86, 2: 365.7},
								  'y': {0: 451.76, 1: 451.76, 2: 451.76},
								  'distance': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
								  'average_speed': {0: 0.0, 1: 0.19999999999998863, 2: 0.160000000000025},
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
		case = compute_average_speed(inp, fps = 1)
		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i], check_dtype=False)

	def test_average_acceleration(self):
		ref = avaccel
		inp = avspeed
		case = compute_average_acceleration(inp, fps = 1)
		for i in ref.keys():
			pd.testing.assert_frame_equal(ref[i], case[i], check_dtype=False)

	def test_computing_stops(self):
		ref = stops
		inp = avaccel
		case = computing_stops(inp, threshold_speed = 0.5)
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
		case_time = extract_features(inp_time, fps=1, stop_threshold=0.1).round(4)

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
		pd.testing.assert_frame_equal(ref,case, check_dtype=False)

	def test_all_ts_features(self):
		ref = pd.DataFrame(ts_feats).round(4)
		inp = pd.DataFrame(feats)
		case = ts_all_features(inp).round(4).iloc[:,15:30]
		ref = ref.rename_axis('variable', axis=1)
		ref = ref.rename_axis("id", axis=0)
		pd.testing.assert_frame_equal(ref,case, check_dtype=False)

	def test_outlier_detection(self):
		ref = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
										  9: 905, 10: 312, 11: 511},
							'outlier': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31, 11: 370.01},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07, 11: 428.85},
							'distance': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447, 7: 0.08,
										 8: 0.4597, 9: 0.2, 10: 0.3, 11: 0.03},
							'average_speed': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447, 7: 0.08,
											  8: 0.4597, 9: 0.2, 10: 0.3, 11: 0.03},
							'average_acceleration': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447,
													 7: 0.08, 8: 0.4597, 9: 0.2, 10: -0.0905, 11: -0.0147},
							'direction': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -87.0643, 6: 63.4349,
										  7: 180.0, 8: 44.1186, 9: 180.0, 10: -90.0, 11: 90.0},
							'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}})

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
							'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1}})
		case = outlier_detection(inp.round(4))

		pd.testing.assert_frame_equal(ref, case, check_dtype=False)

if __name__ == '__main__':
    unittest.main()
