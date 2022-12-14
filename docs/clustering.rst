Spatio-Temporal Clustering
===========

Movekit allows the use of many different clustering algorithms for spatio-temporal clustering of the movement data: dbscan, hdbscan, agglomerative, optics, spectral, affinitypropagation.
This is done by calling the function `clustering` with the required algorithm. Different hyperparameters can be set for the different algorithms, for an exact description of the parameters of each algorithm see: https://github.com/dbvis-ukon/spatio-temporal-clustering/blob/main/src/st_clustering/st_clustering.py.
Additionally there is the function `clustering_with_splits` which uses a splitting-and-merging method (data is partitioned into frames and merged afterwards) to increase performance for large datasets.

**clustering(algorithm, data, **kwargs)**:
    | Clustering of spatio-temporal data.
    | param algorithm: Choose between 'dbscan', 'hdbscan', 'agglomerative', 'optics', 'spectral', 'affinitypropagation'.
    | param data: DataFrame to perform clustering on.
    | return: labels as numpy array where the label in the first position corresponds to the first row of the input data.

**clustering_with_splits(algorithm, data, frame_size, **kwargs)**:
    | Clustering of spatio-temporal data.
    | param algorithm: Choose between dbscan, hdbscan, agglomerative, optics, spectral, affinitypropagation.
    | param data: DataFrame to perform clustering on.
    | param frame_size: the dataset is partitioned into frames and merged afterwards.
    | return: labels as numpy array where the label in the first position corresponds to the first row of the input data.

.. code-block:: python

    # mkit.clustering(algorithm, data, **kwargs)
    labels = mkit.clustering('dbscan', data_norm, eps1=0.05, eps2=10, min_samples=2)
    #OR cluster with the splitting-and-merging method (data is partitioned into frames and merged afterwards).
    labels = mkit.clustering_with_splits('dbscan', data, frame_size=20, eps1=0.05, eps2=10, min_samples=3)
