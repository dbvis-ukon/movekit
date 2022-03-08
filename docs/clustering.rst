Spatio-Temporal Clustering
===========

Movekit allows the use of many different clustering algorithms for spatio-temporal clustering of the movement data: dbscan, hdbscan, agglomerative, kmeans, optics, spectral, affinitypropagation, birch.
This is done by calling the function `clustering` with the required algorithm. Different hyperparameters can be set for the different algorithms, for an exact description of the parameters of each algorithm see: https://github.com/dbvis-ukon/spatio-temporal-clustering/blob/main/src/st_clustering/st_clustering.py.

.. code-block:: python

    # mkit.clustering(algorithm, data, **kwargs)
    labels = mkit.clustering('dbscan', data_norm, eps1=0.05, eps2=10, min_samples=2)
    #OR cluster with the splitting-and-merging method (data is partitioned into frames and merged afterwards).
    labels = mkit.clustering_with_splits('dbscan', data, frame_size=20, eps1=0.05, eps2=10, min_samples=3)
