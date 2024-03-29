import unittest
from src.movekit.clustering import *

class Test_clustering(unittest.TestCase):
	def test_dtw_matrix(self):

		ref = pd.DataFrame({312:
			{312: 0.0, 511: 74.09067953165595, 607: 38.03365928918165, 811: 80.77636691446091,
			 905: 104.2485347931366}, 511: {312: 74.09067953165595, 511: 0.0, 607: 61.20636468753354,
			811: 154.28892643064987, 905: 46.62598844408468}, 607:
			{312: 38.03365928918165, 511: 61.20636468753354, 607: 0.0, 811: 110.74896336324917, 905: 103.84620569989656},
			811: {312: 80.77636691446091, 511: 154.28892643064987, 607: 110.74896336324917, 811: 0.0,
			905: 177.42829599543492}, 905: {312: 104.2485347931366, 511: 46.62598844408468,
			607: 103.84620569989656, 811: 177.42829599543492, 905: 0.0}})

		inp = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76},
							'distance': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
							'average_speed': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
							'average_acceleration': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
							'direction': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}})
		case = dtw_matrix(inp)
		pd.testing.assert_frame_equal(ref,case, check_dtype=False)



	def test_clustering(self):
		inp = {'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3,
						15: 4, 16: 4, 17: 4, 18: 4, 19: 4, 20: 5, 21: 5, 22: 5, 23: 5, 24: 5, 25: 6, 26: 6, 27: 6,
						28: 6, 29: 6},
			   'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905, 10: 312,
							 11: 511, 12: 607, 13: 811, 14: 905, 15: 312, 16: 511, 17: 607, 18: 811, 19: 905, 20: 312,
							 21: 511, 22: 607, 23: 811, 24: 905, 25: 312, 26: 511, 27: 607, 28: 811, 29: 905},
			   'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01, 7: 390.25,
					 8: 445.48, 9: 365.86, 10: 412.19, 11: 369.89, 12: 390.43, 13: 445.05, 14: 366.16, 15: 405.39,
					 16: 370.11, 17: 390.15, 18: 445.58, 19: 365.82, 20: 405.31, 21: 405.41, 22: 390.37, 23: 445.19, 24: 366.05, 25: 405.29,
					 26: 369.96, 27: 390.22, 28: 445.52, 29: 365.82},
			   'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82, 7: 405.89,
					 8: 412.26, 9: 451.76, 10: 441.66, 11: 428.88, 12: 405.79, 13: 411.84, 14: 451.66, 15: 417.47,
					 16: 428.92, 17: 405.79, 18: 412.16, 19: 451.66, 20: 417.82, 21: 417.75, 22: 405.83, 23: 411.96, 24: 451.74, 25: 417.31,
					 26: 428.88, 27: 405.87, 28: 412.23, 29: 451.79}}
		inp = pd.DataFrame(inp)
		ref = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, -1, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0, 2, 3, 4, 0, 1, 2, 3, 4]

		case = list(clustering('dbscan', inp))

		self.assertListEqual(ref, case)

	def test_heading_difference(self):
		ref = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3},
							'animal_id':{0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
										 9: 905, 10: 312},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07},
							'distance': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447, 7: 0.08,
										 8: 0.4597, 9: 0.2, 10: 0.3},
							'average_speed': {0: 0.230171, 1: 0.022361, 2: 0.040000, 3: 0.229837, 4: 0.100000, 5: 0.230171, 6: 0.022361, 7: 0.040000,
											  8: 0.229837, 9: 0.100000, 10: 0.230171},
							'average_acceleration': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
													 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0},
							'direction': {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0), 5: (0.02, -0.39), 6: (0.02, 0.04),
										  7: (-0.08, 0.0), 8: (0.33, 0.32), 9: (-0.2, 0.0), 10: (0.0, -0.3)},
							'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1},
							'x_centroid': {0: 395.364, 1: 395.364, 2: 395.364, 3: 395.364, 4: 395.364, 5: 395.382,
										   6: 395.382, 7: 395.382, 8: 395.382, 9: 395.382, 10: 405.31},
							'y_centroid': {0: 423.226, 1: 423.226, 2: 423.226, 3: 423.226, 4: 423.226, 5: 423.22,
										   6: 423.22, 7: 423.22, 8: 423.22, 9: 423.22, 10: 417.07},
							'medoid': {0: 312, 1: 312, 2: 312, 3: 312, 4: 312, 5: 312, 6: 312, 7: 312, 8: 312, 9: 312,
									   10: 312},
							'distance_to_centroid': {0: 11.331, 1: 25.975, 2: 18.052, 3: 51.049, 4: 40.901, 5: 11.523,
													 6: 25.983, 7: 18.074, 8: 51.283, 9: 41.062, 10: 0.0},
							'centroid_direction': {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0), 5: (0.018, -0.006),
												   6: (0.018, -0.006), 7: (0.018, -0.006), 8: (0.018, -0.006),
												   9: (0.018, -0.006), 10: (9.928, -6.15)},
							'heading_difference': {0: 0.000000, 1: 0.000000, 2: 0.000000, 3: 0.000000, 4: 0.000000, 5: 0.364399,
												   6: 0.141421, 7: -0.948683, 8: 0.460919,
												   9: -0.948683, 10: 0.526608}})
		inp = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
										  9: 905, 10: 312},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07},
							'distance': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.3905, 6: 0.0447, 7: 0.08,
										 8: 0.4597, 9: 0.2, 10: 0.3},
							'average_speed': {0: 0.230171, 1: 0.022361, 2: 0.040000, 3: 0.229837, 4: 0.100000, 5: 0.230171, 6: 0.022361, 7: 0.040000,
											  8: 0.229837, 9: 0.100000, 10: 0.230171},
							'average_acceleration': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
													 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0},
							'direction': {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0), 3: (0.0, 0.0), 4: (0.0, 0.0), 5: (0.02, -0.39), 6: (0.02, 0.04),
										  7: (-0.08, 0.0), 8: (0.33, 0.32), 9: (-0.2, 0.0), 10: (0.0, -0.3)},
							'stopped': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}})
		case = get_heading_difference(inp)
		pd.testing.assert_frame_equal(ref,case)


	def test_polarization(self):
		ref = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905, 10: 312},
							'polarization': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.196415, 6: 0.196415, 7: 0.196415, 8: 0.196415, 9: 0.196415, 10: 1.0}})
		inp = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
										  9: 905, 10: 312},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07}})
		case = compute_polarization(inp).loc[:,['time','animal_id','polarization']]
		pd.testing.assert_frame_equal(ref, case)

	# Might be transformed to a general spatial object test covering all 3 types of spatial objs
	def test_voronoi(self):
		ref_area = pd.DataFrame({'voronoi_volume': {0: 2414.225693024696, 1: float("inf"), 2: float("inf"),
												  3: float("inf"), 4: float("inf"), 5: 2389.8757249973805,
												  6: float("inf"), 7: float("inf"), 8: float("inf"), 9: float("inf")}})
		ref_points = [[[405.29, 417.76],
					  [369.99, 428.78],
					  [390.33, 405.89],
					  [445.15, 411.94],
					  [366.06, 451.76]],
					 [[405.31, 417.37],
					  [370.01, 428.82],
					  [390.25, 405.89],
					  [445.48, 412.26],
					  [365.86, 451.76]]]

		inp = pd.DataFrame({'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2},
							'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811,
										  9: 905},
							'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01,
								  7: 390.25, 8: 445.48, 9: 365.86},
							'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82,
								  7: 405.89, 8: 412.26, 9: 451.76}})

		#area, diagrams = voronoi_diagram(inp)

		#objs = get_spatial_objects(inp)


		# Obtaining all spatial object types
		objs = get_spatial_objects(inp, group_output=False).sort_values(by=['time', 'animal_id'])


		# Generating list of points from diagrams of the two tested time points
		v_objs = objs['voronoi_object']
		pointlst = []
		for i in range(0, len(v_objs), 5):
			pointlst.append(v_objs[i].points.tolist())

		# Extracting the voronoi volume
		area = pd.DataFrame(objs['voronoi_volume'])

		# testing area dataframe
		pd.testing.assert_frame_equal(ref_area, area)

		# testing points lists
		self.assertListEqual(ref_points, pointlst)

if __name__ == '__main__':
    unittest.main()
