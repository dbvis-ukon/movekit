import os
import unittest
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
               'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 421.18, 4: None, 9: 428.82, 7: 412.26, 8: 417.37,
                     5: 451.76, 6: 405.89}}

        inp = pd.DataFrame(missings)
        ref = pd.DataFrame(interpolated)

        # Interpolated with default limit = 1
        case = interpolate(inp)
        pd.testing.assert_frame_equal(ref,case, check_dtype=False)

    def test_preprocess(self):
        up_missings = {
            'time': {0: None, 1: 1, 2: 1, 3: 1, 4: 1, 9: 2, 7: 2, 8: 2, 5: 2, 6: 2},
                    'animal_id': {0: 312, 1 : 511, 2: 607,3: 811,4: 905,9: 511,7: 811,8: 312,5: 905, 6: 607},
                    'x': {0: 405.29, 1: 369.99, 2: None, 3: 445.15, 4: 366.06, 9: 370.01, 7: 445.48, 8: 405.31,
                          5: 365.86, 6: 390.25},
                    'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: None, 4: None, 9: 428.82, 7: 412.26, 8: 417.37,
                          5: 451.76, 6: 405.89}}
        preprocessed = {'time': {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 2.0, 6: 2.0, 7: 2.0, 8: 2.0, 9: 2.0},
                        'animal_id': {1: 511, 2: 607, 3: 811, 4: 905, 5: 905, 6: 607, 7: 811, 8: 312, 9: 511},
                        'x': {1: 369.99, 2: 407.57, 3: 445.15, 4: 366.06, 5: 365.86, 6: 390.25, 7: 445.48, 8: 405.31,
                              9: 370.01}, 'y': {1: 428.78, 2: 405.89, 3: 421.18, 4: None, 5: 451.76, 6: 405.89,
                                                7: 412.26, 8: 417.37, 9: 428.82}}

        inp = pd.DataFrame(up_missings)
        ref = pd.DataFrame(preprocessed)
        case = preprocess(inp)
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

        inp = dict_groups

        ref = dict_replaced
        arr_index = np.array([0,1,2])
        case = replace_parts_animal_movement(inp, 811, arr_index, replacement_value_x = 100, replacement_value_y = 90)
        pd.testing.assert_frame_equal(ref[811],case[811], check_dtype=False)

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

        inp = dict_groups
        ref = resamples
        case = resample_systematic(inp, 1)
        for i in ref.keys():
            self.assertTrue((ref[i].all() == case[i].all()).all(), "Results don't match")


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


        inp = dict_groups
        case = resample_random(inp, 10)
        for i in case.keys():
            self.assertEqual(len(case[i]), 10, "Results don't match")

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
        inp = dict_groups
        ref = split_traj
        case = split_trajectories(inp, 1)
        for i in ref.keys():
            pd.testing.assert_frame_equal(ref[i], case[i])


if __name__ == '__main__':
    unittest.main()
