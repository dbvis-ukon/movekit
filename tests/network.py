import unittest
from src.movekit.network import *


class Test_network(unittest.TestCase):

    def test_network_time_graphlist(self):
        data = pd.DataFrame({
        'time': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 13: 3, 14: 3, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4},
        'animal_id': {0: 312, 1: 511, 2: 607, 3: 811, 4: 905, 5: 312, 6: 511, 7: 607, 8: 811, 9: 905, 10: 312, 11: 511, 12: 607, 13: 811, 14: 905, 15: 312, 16: 511, 17: 607, 18: 811, 19: 905},
        'x': {0: 405.29, 1: 369.99, 2: 390.33, 3: 445.15, 4: 366.06, 5: 405.31, 6: 370.01, 7: 390.25, 8: 445.48, 9: 365.86, 10: 405.31, 11: 370.01, 12: 390.17, 13: 445.77, 14: 365.7, 15: 405.3, 16: 370.01, 17: 390.07, 18: 446.03, 19: 365.57},
        'y': {0: 417.76, 1: 428.78, 2: 405.89, 3: 411.94, 4: 451.76, 5: 417.37, 6: 428.82, 7: 405.89, 8: 412.26, 9: 451.76, 10: 417.07, 11: 428.85, 12: 405.88, 13: 412.61, 14: 451.76, 15: 416.86, 16: 428.86, 17: 405.88, 18: 413.0, 19: 451.76}}
        )

        graph = {'time': 4,
         'x_centroid': 395.396,
         'y_centroid': 423.272,
         'medoid': 312,
         'polarization': 0.341926206925558,
         'total_dist': 0.9189596262265768,
         'mean_speed': 0.06238045143063832,
         'mean_acceleration': 2.0122792321330965e-18,
         'mean_distance_centroid': 29.7782,
         'centroid_direction': (0.004, 0.038)}
        node = {'time': 4,
         'animal_id': 312,
         'x': 405.3,
         'y': 416.86,
         'distance': 0.2102379604162655,
         'average_speed': 0.1051189802081327,
         'average_acceleration': -0.07500000000000284,
         'direction': (-0.01, -0.21),
         'turning': 0.9988681377244376,
         'stopped': 1,
         'x_centroid': 395.396,
         'y_centroid': 423.272,
         'medoid': 312,
         'distance_to_centroid': 11.798}

        edge = {'distance': 40.91249809043683}

        lst = network_time_graphlist(data, fps=2)
        outgraph= lst[3].graph
        outnode = lst[3].nodes[312]
        outedge = lst[3].edges[312, 811]

        self.assertDictEqual(outgraph, graph)
        self.assertDictEqual(outnode, node)
        self.assertDictEqual(outedge, edge)
if __name__ == '__main__':
    unittest.main()
