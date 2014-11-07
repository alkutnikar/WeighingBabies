# coding: utf-8
from pandas import *
import pandas as pd
import numpy as np
from urllib import urlopen
from bokeh.plotting import *
import scipy.special
from matplotlib import pyplot as plt    
print 'Pls wait till the data loads and prints columns'
babyCSV = urlopen("/home/alakshminara/Downloads/2008_births.csv")
DF_baby = read_csv(babyCSV)
columns = DF_baby.columns
print list(DF_baby.columns.values)
DF_sammi = pd.DataFrame

def usr_load_sammi():
	global DF_sammi 
	DF_sammi = pd.DataFrame(DF_baby[(DF_baby['RACEMOM']==1) & (DF_baby['RACEDAD']==1) & 
                                (DF_baby['MAGE'] > 25)  & (DF_baby['BPOUND'] < 20) & (DF_baby['MAGE'] < 50) 
                                & (DF_baby['SEX'] == 2)])
	print 'Mean Birth Weight for SammiDF ',DF_sammi['BPOUND'].mean()
	print 'Median Birth Weight for SammiDF ', DF_sammi['BPOUND'].median()	


def usr_scatter_plot(var1,var2):

       	figure(title="Dataset of Babies similar to Sammi's Baby",
       	x_axis_label = var1,
       	y_axis_label = var2)
	# sample the distribution

	# compute ideal values
	#x = DF_sammi[var1]


	# EXERCISE: output to a static HTML file
	output_file('plot1.html')
	# EXERCISE: turn on plot hold
	hold()
	

	scatter(DF_sammi[var1],DF_sammi[var2], marker="square", color="black")#, title="Dataset of Babies similar to Sammi's Baby",xlabel=var1, ylabel=var2)


	# Move the legend to a better place.
	# Acceptable values: 'top_left', 'top_right', 'bottom_left', and 'bottom_right'

	show()

def usr_histogram_plot(var1):
	hold(False)
       	figure(title="Dataset of Babies similar to Sammi's Baby",
       	x_axis_label = 'Birth Weight (lbs.)')
	# sample the distribution

	mu, sigma = 6.834,1      # NOTE: you can tinker with these values if you like

	# sample the distribution
	measured = np.random.normal(mu, sigma, 1000)
	hist, edges = np.histogram(measured, density=True, bins=200)

	# compute ideal values
	x = DF_sammi[var1]
	pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))


	# EXERCISE: output to a static HTML file
	output_file('plot.html')
	# EXERCISE: turn on plot hold
	hold()

	# Use the `quad` renderer to display the histogram bars.
	quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
	     fill_color="#036565", line_color="#033649",

	     # NOTE: these are only needed on the first renderer

	     title="Dataset of Babies similar to Sammi's Baby",
	     tools=""
	)



	# Move the legend to a better place.
	# Acceptable values: 'top_left', 'top_right', 'bottom_left', and 'bottom_right'
	legend().orientation = "top_left"

	show()



usr_load_sammi()
usr_scatter_plot('GAINED','BPOUND')
usr_histogram_plot('BPOUND')
#print_mean()
