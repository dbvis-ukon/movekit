"""
  Preprocess the data frame
  Author: Arjun Majumdar, Eren Cakmak
  Created: July, 2019
"""


def linear_interpolation(data, threshold):
    '''
        Function to interpolate missing values for 'x' and 'y' attributes
        in dataset.
        'threshold' parameter decides the number of rows till which, data
        should NOT be deleted.
        '''

    # Get indices of missing values for 'x' attribute in a list-
    missing_x_values = list(data[data['x'].isnull()].index)

    # Get indices of missing values for 'y' attribute in a list-
    missing_y_values = list(data[data['y'].isnull()].index)

    print("\nNumber of missing values in 'x' attribute = {0}".format(
        len(missing_x_values)))
    print("Number of missing values in 'y' attribute = {0}\n".format(
        len(missing_y_values)))

    # counter for outer loop-
    i = 0
    # counter for inner loop-
    j = 0
    # start and end counters-
    start = end = 0
    # count length of sequence found-
    k = 1

    # List containing indices to be deleted-
    indices_to_delete = []

    # threshold = 10

    while i < len(missing_x_values) - 1:
        start = end = missing_x_values[i]
        k = 1
        # j = missing_x_values[i]
        j = i

        # print("\ni = {0} & j = {1}".format(i, j))

        while j < (len(missing_x_values) - 1):
            # print("j = ", j)
            if missing_x_values[j] + 1 == missing_x_values[j + 1]:
                k += 1
                j += 1
                # end = j
                end = missing_x_values[j]
            else:
                # i = j + 1
                break

        i = j + 1

        if k >= threshold:
            # Delete rows-
            print("\nDelete sequence from {0} to {1}\n".format(start, end))
            # data = data.drop(data.index[start:end + 1])
            # data.drop(data.index[start: end + 1], inplace = True, axis = 0)
            for x in range(start, end + 1):
                indices_to_delete.append(x)

        elif k > 1:
            # Perform Linear Interpolation-
            print("\nSequence length = {0}. Start = {1} & End = {2}".format(
                k, start, end))

    # Delete indices-
    data_del = data.drop(data.index[indices_to_delete], axis=0)

    return data_del
