import pandas as pd


def boilerbands(stock_data, k, n_adj, moving_average = 'normal'):
	mean = pd.stats.moments.rolling_mean(stock_data, k)
	std = pd.stats.moments.rolling_std(stock_data, k)
	
	top_band = mean + (std*n_adj)
	bot_band = mean - (std*n_adj)


	return mean, top_band, bot_band


	#moving_average = 'normal', 'exponential'
	#return a new frame with boil

def moving_average(stock_data, k):
	return stock_data.rolling_mean(k)