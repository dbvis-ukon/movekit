"""
	Preprocess the Pandas Data Frame by filtering out a
	from and to slice
	Author: Arjun Majumdar, Eren Cakmak
	Created: August, 2019
"""


import pandas as pd


def filter_dataframe(data, frm, to):
	"""
	A function to filter the dataset, which is the first
	argument to the function using 'frm' and 'to' as the
	second and third arguments.
	Please note that both 'frm' and 'to' are included in
	the returned filtered Pandas Data frame.

	Returns a filtered Pandas Data frame according to 'frm'
	and 'to' arguments
	"""

	return data.loc[(data['time'] >= frm) & (data['time'] < to), :]





