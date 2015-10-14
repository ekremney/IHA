"""
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plotpra(precision, area):
	sns.set_style("white")
	sns.set_color_codes()
	plt.figure(figsize=(9,9))
	plt.scatter(precision, area)
	plt.show()

"""

import os
for filename in os.listdir("."):
	if filename.startswith('1000'):
		os.rename(filename, 'img' + filename[4:])

