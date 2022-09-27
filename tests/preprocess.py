import os
import unittest
from pandas import Timestamp
import pandas as pd
import numpy as np
import math
import pickle


from src.movekit.preprocess import *




class TestPreprocess(unittest.TestCase):

    def test_interpolate(self):
        missings = {
            'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 9: 2, 7: 2, 8: 2, 5: 2, 6: 2},
                    'animal_id': {0: 312, 1 : 511, 2: 607,3: 811,4: 905,9: 511,7: 811,8: 312,5: 905, 6: 607},
                    'x': {0: 405.29, 1: 369.99, 2: None, 3: 445.15, 4: 366.06, 9: 370.01, 7: 445.48, 8: 405.31,
                          5: 365.86, 6: 390.25},
                    'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: None, 4: None, 9: 428.82, 7: 412.26, 8: 417.37,
                          5: 451.76, 6: 405.89}}

        interpolated = {
            'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 9: 2, 7: 2, 8: 2, 5: 2, 6: 2},
               'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 9: 511, 7: 811, 8: 312, 5: 905, 6: 607},
               'x': {0: 405.29, 1: 369.99, 2: 407.57, 3: 445.15, 4: 366.06, 9: 370.01, 7: 445.48, 8: 405.31,
                     5: 365.86, 6: 390.25},
               'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 413.53, 4: None, 9: 428.82, 7: 412.26, 8: 417.37,
                     5: 451.76, 6: 405.89}}

        inp = pd.DataFrame(missings)
        ref = pd.DataFrame(interpolated)

        # Interpolated with default limit = 1
        case = interpolate(inp)
        pd.testing.assert_frame_equal(ref,case, check_dtype=False)

    def test_preprocess(self):
        up_missings = {
            'time': {0: None, 1: 1, 2: 1, 3: 1, 4: 1,5: 2, 6: 2, 7: 2, 8: 2, 9: 2},
                    'animal_id': {0: 312, 1 : 511, 2: 607,3: 811,4: 905,5: 905,6: 607,7: 811,8: 312,9: 511},
                    'x': {0: 405.29, 1: 369.99, 2: None, 3: 445.15, 4: 366.06,5: 365.86, 6: 390.25, 7: 445.48, 8: 405.31, 9: 370.01},
                    'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: None, 4: None,5: 451.76, 6: 405.89, 7: 412.26, 8: 417.37, 9: 428.82}}
        preprocessed = {'time': {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0},
                        'animal_id': {1: 511, 2: 607, 3: 811, 4: 905, 5: 905, 6: 607, 7: 811, 8: 312, 9: 511},
                        'x': {1: 369.99, 2: 407.57, 3: 445.15, 4: 366.06, 5: 365.86, 6: 390.25, 7: 445.48, 8: 405.31,
                              9: 370.01}, 'y': {1: 428.78, 2: 405.89, 3: 421.18, 4: None, 5: 451.76, 6: 405.89,
                                                7: 412.26, 8: 417.37, 9: 428.82}}

        inp = pd.DataFrame(up_missings)
        ref = pd.DataFrame(preprocessed)
        case = preprocess(inp, interpolation=True)
        pd.testing.assert_frame_equal(ref, case, check_dtype=False)

    def test_preprocess_time(self):
        dat_time={
            'time': {0: Timestamp('2020-03-27 11:57:07'), 1: Timestamp('2020-03-27 11:57:09'),
                     2: Timestamp('2020-03-27 11:57:11'), 3: Timestamp('2020-03-27 11:57:13'),
                     4: Timestamp('2020-03-27 11:57:15'), 5: Timestamp('2020-03-27 11:57:17'),
                     6: Timestamp('2020-03-27 11:57:19'), 7: Timestamp('2020-03-27 11:57:21'),
                     8: Timestamp('2020-03-27 11:57:23'), 9: Timestamp('2020-03-27 11:57:25'),
                     10: Timestamp('2020-03-27 11:57:27'), 11: Timestamp('2020-03-27 11:57:29'),
                     12: Timestamp('2020-03-27 11:57:31'), 13: Timestamp('2020-03-27 11:57:33'),
                     14: Timestamp('2020-03-27 11:57:35'), 15: Timestamp('2020-03-27 11:57:37'),
                     16: Timestamp('2020-03-27 11:57:39'), 17: Timestamp('2020-03-27 11:57:41'),
                     18: Timestamp('2020-03-27 11:57:43'), 19: Timestamp('2020-03-27 11:57:45'),
                     20: Timestamp('2020-03-27 11:57:47'), 21: Timestamp('2020-03-27 11:57:49'),
                     22: Timestamp('2020-03-27 11:57:51'), 23: Timestamp('2020-03-27 11:57:53'),
                     24: Timestamp('2020-03-27 11:57:55'), 25: Timestamp('2020-03-27 11:57:57'),
                     26: Timestamp('2020-03-27 11:57:59'), 27: Timestamp('2020-03-27 11:58:01'),
                     28: Timestamp('2020-03-27 11:58:03'), 29: Timestamp('2020-03-27 11:58:05'),
                     30: Timestamp('2020-03-27 11:58:07'), 31: Timestamp('2020-03-27 11:58:09'),
                     32: Timestamp('2020-03-27 11:58:11'), 33: Timestamp('2020-03-27 11:58:13'),
                     34: Timestamp('2020-03-27 11:58:15'), 35: Timestamp('2020-03-27 11:58:17'),
                     36: Timestamp('2020-03-27 11:58:19'), 37: Timestamp('2020-03-27 11:58:21'),
                     38: Timestamp('2020-03-27 11:58:23'), 39: Timestamp('2020-03-27 11:58:25'),
                     40: Timestamp('2020-03-27 11:58:27'), 41: Timestamp('2020-03-27 11:58:29'),
                     42: Timestamp('2020-03-27 11:58:31'), 43: Timestamp('2020-03-27 11:58:33'),
                     44: Timestamp('2020-03-27 11:58:35'), 45: Timestamp('2020-03-27 11:58:37'),
                     46: Timestamp('2020-03-27 11:58:39'), 47: Timestamp('2020-03-27 11:58:41'),
                     48: Timestamp('2020-03-27 11:58:43'), 49: Timestamp('2020-03-27 11:58:45')},
            'animal_id': {0: 312.0, 1: 459.5, 2: 607.0, 3: 811.0, 4: 905.0, 5: 312.0, 6: 511.0, 7: 607.0,
                          8: 811.0, 9: 905.0, 10: 905.0, 11: 811.0, 12: 312.0, 13: 511.0, 14: 607.0,
                          15: 312.0, 16: 511.0, 17: 607.0, 18: 811.0, 19: 905.0, 20: 811.0, 21: 607.0,
                          22: 905.0, 23: 312.0, 24: 511.0, 25: 312.0, 26: 511.0, 27: 607.0, 28: 811.0,
                          29: 905.0, 30: 312.0, 31: 511.0, 32: 607.0, 33: 811.0, 34: 905.0, 35: 905.0,
                          36: 811.0, 37: 511.0, 38: 312.0, 39: 607.0, 40: 312.0, 41: 511.0, 42: 607.0,
                          43: 811.0, 44: 905.0, 45: 811.0, 46: 312.0, 47: 511.0, 48: 607.0, 49: 905.0},
            'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: None, 4: None, 5: 405.31, 6: 370.01,
                  7: 390.25, 8: 445.48, 9: 365.86, 10: 365.7, 11: 445.77, 12: 405.31, 13: 370.01,
                  14: 390.17, 15: 405.3, 16: 370.01, 17: 390.07, 18: 446.03, 19: 365.57, 20: 446.24,
                  21: 389.98, 22: 365.47, 23: 405.29, 24: 369.99, 25: 405.27, 26: 369.98, 27: 389.88,
                  28: 446.42, 29: 365.39, 30: 405.27, 31: 369.97, 32: 389.79, 33: 446.54, 34: 365.35,
                  35: 365.33, 36: 446.62, 37: 369.97, 38: 405.27, 39: 389.7, 40: 405.31, 41: 369.98,
                  42: 389.68, 43: 446.63, 44: 365.34, 45: 446.55, 46: 405.38, 47: 370.08, 48: 389.62,
                  49: 365.36},
            'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: None,
                  7: 405.89, 8: 412.26, 9: 451.76, 10: 451.76, 11: 412.61, 12: 417.07, 13: 428.85,
                  14: 405.88, 15: 416.86, 16: 428.86, 17: 405.88, 18: 413.0, 19: 451.76,
                  20: 413.42, 21: 405.87, 22: 451.76, 23: 416.71, 24: 428.86, 25: 416.61,
                  26: 428.84, 27: 405.87, 28: 413.86, 29: 451.76, 30: 416.54, 31: 428.82,
                  32: 405.87, 33: 414.34, 34: 451.76, 35: 451.76, 36: 414.85, 37: 428.8,
                  38: 416.49, 39: 405.88, 40: 416.37, 41: 428.82, 42: 405.9, 43: 415.42,
                  44: 451.76, 45: 415.92, 46: 416.27, 47: 428.81, 48: 405.93, 49: 451.81}}

        dat_time_preprocessed = {
            'time': {0: Timestamp('2020-03-27 11:57:07'), 1: Timestamp('2020-03-27 11:57:09'),
                     2: Timestamp('2020-03-27 11:57:11'), 3: Timestamp('2020-03-27 11:57:13'),
                     4: Timestamp('2020-03-27 11:57:15'), 5: Timestamp('2020-03-27 11:57:17'),
                     6: Timestamp('2020-03-27 11:57:19'), 7: Timestamp('2020-03-27 11:57:21'),
                     8: Timestamp('2020-03-27 11:57:23'), 9: Timestamp('2020-03-27 11:57:25'),
                     10: Timestamp('2020-03-27 11:57:27'), 11: Timestamp('2020-03-27 11:57:29'),
                     12: Timestamp('2020-03-27 11:57:31'), 13: Timestamp('2020-03-27 11:57:33'),
                     14: Timestamp('2020-03-27 11:57:35'), 15: Timestamp('2020-03-27 11:57:37'),
                     16: Timestamp('2020-03-27 11:57:39'), 17: Timestamp('2020-03-27 11:57:41'),
                     18: Timestamp('2020-03-27 11:57:43'), 19: Timestamp('2020-03-27 11:57:45'),
                     20: Timestamp('2020-03-27 11:57:47'), 21: Timestamp('2020-03-27 11:57:49'),
                     22: Timestamp('2020-03-27 11:57:51'), 23: Timestamp('2020-03-27 11:57:53'),
                     24: Timestamp('2020-03-27 11:57:55'), 25: Timestamp('2020-03-27 11:57:57'),
                     26: Timestamp('2020-03-27 11:57:59'), 27: Timestamp('2020-03-27 11:58:01'),
                     28: Timestamp('2020-03-27 11:58:03'), 29: Timestamp('2020-03-27 11:58:05'),
                     30: Timestamp('2020-03-27 11:58:07'), 31: Timestamp('2020-03-27 11:58:09'),
                     32: Timestamp('2020-03-27 11:58:11'), 33: Timestamp('2020-03-27 11:58:13'),
                     34: Timestamp('2020-03-27 11:58:15'), 35: Timestamp('2020-03-27 11:58:17'),
                     36: Timestamp('2020-03-27 11:58:19'), 37: Timestamp('2020-03-27 11:58:21'),
                     38: Timestamp('2020-03-27 11:58:23'), 39: Timestamp('2020-03-27 11:58:25'),
                     40: Timestamp('2020-03-27 11:58:27'), 41: Timestamp('2020-03-27 11:58:29'),
                     42: Timestamp('2020-03-27 11:58:31'), 43: Timestamp('2020-03-27 11:58:33'),
                     44: Timestamp('2020-03-27 11:58:35'), 45: Timestamp('2020-03-27 11:58:37'),
                     46: Timestamp('2020-03-27 11:58:39'), 47: Timestamp('2020-03-27 11:58:41'),
                     48: Timestamp('2020-03-27 11:58:43'), 49: Timestamp('2020-03-27 11:58:45')},
            'animal_id': {0: 312.0, 1: 459.5, 2: 607.0, 3: 811.0, 4: 905.0, 5: 312.0, 6: 511.0, 7: 607.0,
                          8: 811.0, 9: 905.0, 10: 905.0, 11: 811.0, 12: 312.0, 13: 511.0, 14: 607.0,
                          15: 312.0, 16: 511.0, 17: 607.0, 18: 811.0, 19: 905.0, 20: 811.0, 21: 607.0,
                          22: 905.0, 23: 312.0, 24: 511.0, 25: 312.0, 26: 511.0, 27: 607.0, 28: 811.0,
                          29: 905.0, 30: 312.0, 31: 511.0, 32: 607.0, 33: 811.0, 34: 905.0, 35: 905.0,
                          36: 811.0, 37: 511.0, 38: 312.0, 39: 607.0, 40: 312.0, 41: 511.0, 42: 607.0,
                          43: 811.0, 44: 905.0, 45: 811.0, 46: 312.0, 47: 511.0, 48: 607.0, 49: 905.0},
            'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 395.3233333333333, 4: None, 5: 405.31, 6: 370.01,
                  7: 390.25, 8: 445.48, 9: 365.86, 10: 365.7, 11: 445.77, 12: 405.31, 13: 370.01,
                  14: 390.17, 15: 405.3, 16: 370.01, 17: 390.07, 18: 446.03, 19: 365.57, 20: 446.24,
                  21: 389.98, 22: 365.47, 23: 405.29, 24: 369.99, 25: 405.27, 26: 369.98, 27: 389.88,
                  28: 446.42, 29: 365.39, 30: 405.27, 31: 369.97, 32: 389.79, 33: 446.54, 34: 365.35,
                  35: 365.33, 36: 446.62, 37: 369.97, 38: 405.27, 39: 389.7, 40: 405.31, 41: 369.98,
                  42: 389.68, 43: 446.63, 44: 365.34, 45: 446.55, 46: 405.38, 47: 370.08, 48: 389.62,
                  49: 365.36},
            'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 411.63,
                  7: 405.89, 8: 412.26, 9: 451.76, 10: 451.76, 11: 412.61, 12: 417.07, 13: 428.85,
                  14: 405.88, 15: 416.86, 16: 428.86, 17: 405.88, 18: 413.0, 19: 451.76,
                  20: 413.42, 21: 405.87, 22: 451.76, 23: 416.71, 24: 428.86, 25: 416.61,
                  26: 428.84, 27: 405.87, 28: 413.86, 29: 451.76, 30: 416.54, 31: 428.82,
                  32: 405.87, 33: 414.34, 34: 451.76, 35: 451.76, 36: 414.85, 37: 428.8,
                  38: 416.49, 39: 405.88, 40: 416.37, 41: 428.82, 42: 405.9, 43: 415.42,
                  44: 451.76, 45: 415.92, 46: 416.27, 47: 428.81, 48: 405.93, 49: 451.81}}

        inp = pd.DataFrame(dat_time)
        ref = pd.DataFrame(dat_time_preprocessed)
        case = preprocess(inp, interpolation=True, date_format=True)
        pd.testing.assert_frame_equal(ref, case, check_dtype=False)

    def test_filter_dataframe(self):
        records = {
            "time": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            "animal_id": [312, 511, 607, 811, 905, 511, 811, 312, 905, 607, 312, 511, 607, 811, 905],
            "x": [405.29, 369.99, 390.33, 445.15, 366.06, 370.01, 445.48, 405.31, 365.86, 390.25, 405.31, 370.01,
                  390.17, 445.77, 365.7],
            "y": [417.76, 428.78, 405.89, 411.94, 451.76, 428.82, 412.26, 417.37, 451.76, 405.89, 417.07, 428.85,
                  405.88, 412.61, 451.76]}

        filtered = {
            'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3,
                             14: 3},
                    'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 905, 6: 607, 7: 811, 8: 312, 9: 511,
                                  10: 312, 11: 511, 12: 607, 13: 811, 14: 905},
                    'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 365.86, 6: 390.25, 7: 445.48,
                          8: 405.31, 9: 370.01, 10: 405.31, 11: 370.01, 12: 390.17, 13: 445.77, 14: 365.7},
                    'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 451.76, 6: 405.89, 7: 412.26,
                          8: 417.37, 9: 428.82, 10: 417.07, 11: 428.85, 12: 405.88, 13: 412.61, 14: 451.76}}

        inp = pd.DataFrame(records)
        ref = pd.DataFrame(filtered).sort_values(by=["time", "animal_id"]).reset_index(drop=True)
        case = filter_dataframe(inp, 1,3).sort_values(by=["time", "animal_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(ref,case, check_dtype=False)

    def test_replace_parts_movement(self):
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

        dict_replaced = {
            312: pd.DataFrame({'time': {0: 1, 1: 2, 2: 3},
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
            'x': {0: 100.00, 1: 100.00, 2: 100.00},
            'y': {0: 90.00, 1: 90.00, 2: 90.00},
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

        inp = pd.DataFrame()
        for key in dict_groups.keys():
            inp = pd.concat([inp, dict_groups[key]])

        ref = pd.DataFrame()
        for key in dict_replaced.keys():
            ref = pd.concat([ref, dict_replaced[key]], ignore_index=True)
        ref.sort_values(['time', 'animal_id'], ascending=True, inplace=True)
        ref.reset_index(drop=True, inplace=True)

        arr_index = np.array([1,2,3])
        case = replace_parts_animal_movement(inp, 811, arr_index, replacement_value_x = 100, replacement_value_y = 90)
        pd.testing.assert_frame_equal(ref,case, check_dtype=False)

    def test_resample_systematic(self):
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

        resamples = {
        312: pd.DataFrame({
            'time': {0: 1},
            'animal_id': {0: 312},
            'x': {0: 405.29},
            'y': {0: 417.76},
            'distance': {0: None},
            'average_speed': {0: None},
            'average_acceleration': {0: None},
            'direction': {0: None},
            'stopped': {0: None}}),
        511: pd.DataFrame({
            'time': {0: 1},
            'animal_id': {0: 511},
            'x': {0: 369.99},
            'y': {0: 428.78},
            'distance': {0: None},
            'average_speed': {0: None},
            'average_acceleration': {0: None},
            'direction': {0: None},
            'stopped': {0: None}}),
        607: pd.DataFrame({
            'time': {0: 1},
            'animal_id': {0: 607},
            'x': {0: 390.33},
            'y': {0: 405.89},
            'distance': {0: None},
            'average_speed': {0: None},
            'average_acceleration': {0: None},
            'direction': {0: None},
            'stopped': {0: None}}),
        811: pd.DataFrame({
            'time': {0: 1},
            'animal_id': {0: 811},
            'x': {0: 445.15},
            'y': {0: 411.94},
            'distance': {0: None},
            'average_speed': {0: None},
            'average_acceleration': {0: None},
            'direction': {0: None},
            'stopped': {0: None}}),
        905: pd.DataFrame({
            'time': {0: 1},
            'animal_id': {0: 905},
            'x': {0: 366.06},
            'y': {0: 451.76},
            'distance': {0: None},
            'average_speed': {0: None},
            'average_acceleration': {0: None},
            'direction': {0: None},
            'stopped': {0: None}})}

        inp = pd.DataFrame()
        for key in dict_groups.keys():
            inp = pd.concat([inp, dict_groups[key]])

        ref = pd.DataFrame()
        for key in resamples.keys():
            ref = pd.concat([ref, resamples[key]], ignore_index=True)

        case = resample_systematic(inp, 1)

        pd.testing.assert_frame_equal(ref, case, check_dtype=False)


    def test_resample_random(self):
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

        inp = pd.DataFrame()
        for key in dict_groups.keys():
            inp = pd.concat([inp, dict_groups[key]])

        case = resample_random(inp, 10)
        for i in dict_groups.keys():
            self.assertEqual(len(case[case['animal_id'] == i]), 10, "Results don't match")

    def test_split_trajectories(self):
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
        split_traj  = {
                    "group_312_df1":pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
                                                   'animal_id': {0: 312, 1: 312, 2: 312},
                                                   'x': {0: 405.29, 1: 405.31, 2: 405.31},
                                                   'y': {0: 417.76, 1: 417.37, 2: 417.07},
                                                   'distance': {0: None, 1: None, 2: None},
                                                   'average_speed': {0: None, 1: None, 2: None},
                                                   'average_acceleration': {0: None, 1: None, 2: None},
                                                   'direction': {0: None, 1: None, 2: None},
                                                   'stopped': {0: None, 1: None, 2: None}} ),
                    "group_511_df1":pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
                                                   'animal_id': {0: 511, 1: 511, 2: 511},
                                                   'x': {0: 369.99, 1: 370.01, 2: 370.01},
                                                   'y': {0: 428.78, 1: 428.82, 2: 428.85},
                                                   'distance': {0: None, 1: None, 2: None},
                                                   'average_speed': {0: None, 1: None, 2: None},
                                                   'average_acceleration': {0: None, 1: None, 2: None},
                                                   'direction': {0: None, 1: None, 2: None},
                                                   'stopped': {0: None, 1: None, 2: None}} ),
                    "group_607_df1":pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
                                                   'animal_id': {0: 607, 1: 607, 2: 607},
                                                   'x': {0: 390.33, 1: 390.25, 2: 390.17},
                                                   'y': {0: 405.89, 1: 405.89, 2: 405.88},
                                                   'distance': {0: None, 1: None, 2: None},
                                                   'average_speed': {0: None, 1: None, 2: None},
                                                   'average_acceleration': {0: None, 1: None, 2: None},
                                                   'direction': {0: None, 1: None, 2: None},
                                                   'stopped': {0: None, 1: None, 2: None}} ),
                    "group_811_df1":pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
                                                   'animal_id': {0: 811, 1: 811, 2: 811},
                                                   'x': {0: 445.15, 1: 445.48, 2: 445.77},
                                                   'y': {0: 411.94, 1: 412.26, 2: 412.61},
                                                   'distance': {0: None, 1: None, 2: None},
                                                   'average_speed': {0: None, 1: None, 2: None},
                                                   'average_acceleration': {0: None, 1: None, 2: None},
                                                   'direction': {0: None, 1: None, 2: None},
                                                   'stopped': {0: None, 1: None, 2: None}} ),
                    "group_905_df1":pd.DataFrame( {'time': {0: 1, 1: 2, 2: 3},
                                                   'animal_id': {0: 905, 1: 905, 2: 905},
                                                   'x': {0: 366.06, 1: 365.86, 2: 365.7},
                                                   'y': {0: 451.76, 1: 451.76, 2: 451.76},
                                                   'distance': {0: None, 1: None, 2: None},
                                                   'average_speed': {0: None, 1: None, 2: None},
                                                   'average_acceleration': {0: None, 1: None, 2: None},
                                                   'direction': {0: None, 1: None, 2: None},
                                                   'stopped': {0: None, 1: None, 2: None}} )}
        inp = pd.DataFrame()
        for key in dict_groups.keys():
            inp = pd.concat([inp, dict_groups[key]])
        ref = split_traj
        case = split_trajectories(inp, 1)
        for i in ref.keys():
            pd.testing.assert_frame_equal(ref[i], case[i])

    def test_convert_measures(self):
        ref = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2},
               'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905},
               'x': {0: 0.4952, 1: 0.0519, 2: 0.3073, 3: 0.9959, 4: 0.0025, 5: 0.4955, 6: 0.0521, 7: 0.3063, 8: 1.0,
                     9: 0.0},
               'y': {0: 0.2588, 1: 0.499, 2: 0.0, 3: 0.1319, 4: 1.0, 5: 0.2503, 6: 0.4999, 7: 0.0, 8: 0.1389, 9: 1.0}})

        inp = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2},
                            'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
                                          9: 905},
                            'x': {0: 0.49522732981662904, 1: 0.05187138909821647, 2: 0.3073348404923382,
                                  3: 0.995855312735493, 4: 0.0025119316754582846, 5: 0.49547852298417466,
                                  6: 0.05212258226576208, 7: 0.30633006782215505, 8: 1.0, 9: 0.0},
                            'y': {0: 0.25877479834314376, 1: 0.49901896664486556, 2: 0.0, 3: 0.13189448441247026,
                                  4: 1.0, 5: 0.2502725092653154, 6: 0.49989099629387407, 7: 0.0, 8: 0.13887072160453465,
                                  9: 1.0}})

        case = convert_measueres(inp).round(4)

        pd.testing.assert_frame_equal(ref, case)
if __name__ == '__main__':
    unittest.main()
