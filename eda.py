import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from matplotlib.finance import candlestick
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
import seaborn as sns
import time
import random
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller as adf

from technicaltools import utils

all_data = pd.read_csv('WIKI_PRICES.csv',delimiter=',')
all_data['date_num'] = all_data['date'].apply(lambda d: date2num(datetime.datetime.strptime(d, "%Y-%m-%d")))

#create required normalization columns
#this can be in its own function... some day
t0 = all_data.groupby('ticker').first()[['adj_close']]
t0.columns = ['adj_close_t0']
t0_df = pd.DataFrame(t0)
t0_df.reset_index(inplace = True)

#left join t0 on ticker
all_data = pd.merge(all_data, t0_df, on = 'ticker', how = 'left')

#Normalize data
all_data['norm_close'] = all_data['adj_close'] / all_data['adj_close_t0']
all_data['ln_close'] = np.log(all_data['norm_close'])

tickers = all_data['ticker'].unique()


#easy plot stocks across provided tickers and columns to plot ie close, adj_close, normalized close
def plot_stocks(stock_data, tickers, column, title = "Stock Performace"):
	for ticker in tickers:
		stock_points = stock_data.loc[stock_data['ticker'] == ticker][['date_num', column]]
		plt.plot(stock_points['date_num'], stock_points[column], label = ticker)

	plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))
	#plt.gca().xaxis.set_major_locator(DayLocator())


	#This was used specically to annotate on graph for my blog post.
	#This test was run with GOOGL, MSFT pair
	#This adds arrows to the chart
	if False == True:
		##PLOT ANNOTATIONS FOR MICROSOFT SPECIFICALLY
		plt.annotate('Divergence', xy = (735536, 1.36), xytext = (735294, 1.65), arrowprops=dict(facecolor='yellow', shrink=0.05))
		plt.annotate('', xy = (735522, 0.92), xytext = (735408, 1.65), arrowprops=dict(facecolor='yellow', shrink=0.05))

		plt.annotate('Converge', xy = (735631, 1.10), xytext = (735713, 0.85), arrowprops=dict(facecolor='pink', shrink=0.05))
		#plt.annotate('', xy = (735522, 0.92), xytext = (735408, 1.65), arrowprops=dict(facecolor='pink', shrink=0.05))

		##PLOT ANNOTATIONS FOR MICROSOFT SPECIFICALLY
		plt.annotate('Divergence', xy = (735715, 1.36), xytext = (735744, 1.6), arrowprops=dict(facecolor='yellow', shrink=0.05))
		#plt.annotate('', xy = (735727, 0.92), xytext = (735798, 1.45), arrowprops=dict(facecolor='yellow', shrink=0.05))
		
		plt.annotate('Converge', xy = (735795, 1.22), xytext = (735991, 1.00), arrowprops=dict(facecolor='pink', shrink=0.05))
		#plt.annotate('', xy = (735522, 0.92), xytext = (735408, 1.65), arrowprops=dict(facecolor='pink', shrink=0.05))
	

	plt.xlabel('Date')
	plt.ylabel('% Gains')
	plt.title(title)
	plt.legend()
	plt.show()

def run_ADF_regression(stock_data, t1, t2, column = 'adj_close'):
	x_points = stock_data.loc[stock_data['ticker'] == t1][column]
	y_points = stock_data.loc[stock_data['ticker'] == t2][column]

	reg = linear_model.LinearRegression()
	reg.fit(x_points.reshape((len(x_points), 1)), y_points)

	return reg

def plot_scatter_compare(stock_data, t1, t2, column = 'adj_close', with_regression = False, plot_title = 'Scatter Regression'):
	x_points = stock_data.loc[stock_data['ticker'] == t1][column]
	y_points = stock_data.loc[stock_data['ticker'] == t2][column]

	if with_regression:
		reg = run_ADF_regression(stock_data, t1, t2, column)

		# print "Coefficients: \n" , reg.coef_
		# print "Intercept: \n" , reg.intercept_
		# print "Residues: \n" , reg.residues_
		plt.plot(x_points, reg.predict(x_points.reshape((len(x_points), 1))), c = 'red')

	plt.xlabel(t1)
	plt.ylabel(t2)
	plt.scatter(x_points, y_points)
	plt.title(plot_title)
	plt.show()

def get_residuals(stock_data, t1, t2, column = 'adj_close'):
	stock_a = stock_data.loc[stock_data['ticker'] == t1][['date_num', column]]
	stock_b = stock_data.loc[stock_data['ticker'] == t2][['date_num', column]]

	stock_a.reset_index(inplace = True)
	stock_b.reset_index(inplace = True)

	reg = run_ADF_regression(stock_data, t1, t2, column)
	residuals = reg.intercept_ + reg.coef_[0] * stock_b[column] - stock_a[column]

	res_df = residuals.to_frame()
	res_df.columns = ['residuals']

	res_df['date_num'] = stock_a['date_num']

	return res_df


def ADF_test(residuals, output_log = False, title = "ADF Test Results"):
	t0 = residuals
	t1 = residuals.shift()

	shifted = t1 - t0
	shifted.dropna(inplace = True)

	plt.plot(shifted, c='green')
	plt.show()

	adf_value = adf(shifted, regression = 'nc')

	test_statistic = adf_value[0]
	pvalue = adf_value[1]
	usedlags = adf_value[2]
	nobs = adf_value[3]


	if output_log:
		#output on figure eventually, that looks really professional
		print title
		print "Test Statistic: %.4f\nP-Value: %.4f\nLags Used: %d\nObservations: %d" % (test_statistic, pvalue, usedlags, nobs)

		for crit in adf_value[4]:
			print crit, adf_value[4][crit]
			#print "Critical Value (%s): %.3f" % (crit, adf_value[crit])

	return adf_value


def plot_residuals(stock_data, t1, t2, column = 'adj_close', plot_title = 'residuals'):

		residuals = get_residuals(stock_data, t1, t2, column)
		
		x = range(0, len(residuals))
		plt.plot(residuals['date_num'], residuals['residuals'], c = 'red')

		#mean, top, bot = utils.boilerbands(residuals, 75, 1.5)
		#x_boiler = range(0, len(mean))

		res_mean = residuals['residuals'].mean()
		res_std = residuals['residuals'].std()

		#plt.annotate('MSFT Overvalued', xy = (735544, 0.59), xytext = (735225, 0.675), arrowprops=dict(facecolor='green', shrink=0.05))
		#plt.annotate('', xy = (735720, 0.56), xytext = (735490, 0.675), arrowprops=dict(facecolor='green', shrink=0.05))
		
		#Apply the standard deviation filter on top of the residuals chart
		#comment out if you dont want the residuals
		plt.plot(residuals['date_num'], np.full((len(residuals), 1), res_mean), c = '#1E293D', linestyle = '--', label = 'Mean (%.3f)' % residuals['residuals'].mean())
		plt.plot(residuals['date_num'], np.full((len(residuals), 1), res_mean + .5*res_std), c = '#445E76', linestyle = '--', label = 'Mean (%.3f)' % residuals['residuals'].mean())
		plt.plot(residuals['date_num'], np.full((len(residuals), 1), res_mean - .5*res_std), c = '#445E76', linestyle = '--', label = 'Mean (%.3f)' % residuals['residuals'].mean())
		plt.plot(residuals['date_num'], np.full((len(residuals), 1), res_mean + res_std), c = '#7498B7', linestyle = '--', label = 'Mean (%.3f)' % residuals['residuals'].mean())
		plt.plot(residuals['date_num'], np.full((len(residuals), 1), res_mean - res_std), c = '#7498B7', linestyle = '--', label = 'Mean (%.3f)' % residuals['residuals'].mean())
		plt.plot(residuals['date_num'], np.full((len(residuals), 1), res_mean + 1.5*res_std), c = '#B1CCE0', linestyle = '--', label = 'Mean (%.3f)' % residuals['residuals'].mean())
		plt.plot(residuals['date_num'], np.full((len(residuals), 1), res_mean - 1.5*res_std), c = '#B1CCE0', linestyle = '--', label = 'Mean (%.3f)' % residuals['residuals'].mean())

		offset_y_frac = 0.012
		offset_y = 0.01

		plt.text(736342, res_mean - offset_y, "$\mu$")
		plt.text(736342, res_mean + 0.5*res_std - offset_y_frac, '$\\frac{1}{2} \sigma$')
		plt.text(736342, res_mean - 0.5*res_std - offset_y_frac, '$\\frac{1}{2} \sigma$')
		plt.text(736342, res_mean + res_std - offset_y, '$\sigma$')
		plt.text(736342, res_mean - res_std - offset_y, '$\sigma$')
		plt.text(736342, res_mean + 1.5*res_std - offset_y_frac, '$\\frac{3}{2} \sigma$')
		plt.text(736342, res_mean - 1.5*res_std - offset_y_frac, '$\\frac{3}{2} \sigma$')
		##END STD OVERLAY

		plt.xlabel('Date')
		plt.ylabel('$S_t$')
		plt.gca().xaxis.set_major_formatter(DateFormatter('%m/%d/%Y'))

		#plt.plot(x_boiler, mean, c='cyan')
		#plt.plot(x_boiler, top, c='pink')
		#plt.plot(x_boiler, bot, c='pink')
		#plt.fill_between(x_boiler, top, bot, alpha = 0.5)
		#plt.legend(frameon = True).get_frame().set_facecolor('white')
		plt.title(plot_title)
		plt.show()



#get a tuple list of all correlations that meet the thresholf found in the corr matrix
#currently it returns a pair for each we may want to address this
#
#Also, we need to filter out and account for low volume stocks as well.
#Probably do that later
def correlation_threshold(corr_matrix, threshold):
	correlation_threshold = threshold
	top_correlations = []

	for index, row in corr_matrix.iterrows():

		for ind, element in enumerate(row):
			if element > correlation_threshold and index != corr_matrix.columns.values[ind]:
				correlation_element = (index, corr_matrix.columns.values[ind], element)
				top_correlations.append(correlation_element)
	
	print top_correlations


def get_correlation(stock_data, tickers, plot_title = 'Covariance Matrix', plot = True, output_log = False, annot = False):
	start_time = time.time()
	index_frames = []
	for symbol in tickers:
		a_s = stock_data.loc[stock_data['ticker'] == symbol][['adj_close', 'date_num']]
		as_index = a_s.set_index(['date_num'])
		as_index.columns = [symbol]
		index_frames.append(as_index)

	joined_data = pd.concat(index_frames, axis = 1)

	#return matrix? print matplot lib
	corr_matrix = joined_data.corr()

	if plot:
		f, ax = plt.subplots(figsize=(11, 9))
		cmap = sns.diverging_palette(220, 10, as_cmap = True)

		# Generate a mask for the upper triangle
		#mask = np.zeros_like(corr_matrix, dtype=np.bool)
		#mask[np.triu_indices_from(mask)] = True

		sns.heatmap(corr_matrix, cmap = cmap, ax = ax, vmax = 1.0, vmin = -1.0, linewidths = 0.5, annot = annot)
		plt.title(plot_title)
		plt.show()


	if output_log:
		print corr_matrix
		print "Run Time: %s seconds" % (time.time() - start_time)

	return corr_matrix



tech_stocks = ['AAPL', 'GOOGL', 'GOOG', 'MSFT', 'FB', 'IBM', 'CSCO']
# auto_stocks = ['F', 'GM', 'MMM', 'ABT', 'ABBV', 'FOX', 'FOXA']
consumer = ['AN', 'AZO', 'CCL', 'CBS', 'CMG', 'COH', 'CMCSA']
portfolio = ['MAS', 'AZO', 'CSCO', 'SJM']

consumer.extend(tech_stocks)

#get_correlation shows the big table with annotations = True
#get_correlation(all_data, consumer, annot = True, plot = True)



stock_set = ['CMCSA', 'CSCO']
plot_stocks(all_data, stock_set, column = 'norm_close', title = "%s / %s" % (stock_set[0], stock_set[1]))
plot_residuals(all_data, stock_set[0], stock_set[1], column = 'norm_close', plot_title = '$S_t$, Stock A: %s  Stock B: %s' % (stock_set[0], stock_set[1]))


print(len(stock_set[1]), len(stock_set[0]))

#plot_scatter_compare(all_data, pairs_test[0], pairs_test[1], 'ln_close', with_regression = True, plot_title="Lognormal Scatter")

#adf_results = ADF_test(get_residuals(all_data, pairs_test[0], pairs_test[1], column = 'ln_close'), output_log = True, title = '%s, %s Lognormal ADF Stationality Test' % (pairs_test[0], pairs_test[1]))
#plot_residuals(all_data, pairs_test[0], pairs_test[1], column = 'ln_close', plot_title = "Residuals, Stock A: %s, Stock B: %s" % (pairs_test[0], pairs_test[1]))



#testing for similiarity
#with this method you will pretty much always get a stationary result. Especially because of the drift factor.
#This means that your results from this test are usless to the underlying data
#can use in tandem with our correlation calculations to hand pick, but I don't think there is an automated system based on this ADF testing

#The Thesis paper mentions Vidyamurthy's paper in which he analyzes the amount of times the residuals cross the mean
#I wonder how this performed, i would say much better at detecting similarities than this ADF. At least for stock moves
#Because they are inhereitely normally disributed, and you will miss the drift if you include a drift factor.

#next step is to create the buy signlals, maybe move some of this code into functions or even modules. I think right now its fine
#eventually maybe have a module for this Cointegration Test and another for the various other types of statistical tests. 




#benchmark against market

def benchmark(stock_data, t1, t2):
	stock_a = stock_data.loc[stock_data['ticker'] == t1][['date_num', 'adj_close']]
	stock_b = stock_data.loc[stock_data['ticker'] == t2][['date_num', 'adj_close']]

	stock_a.reset_index(inplace = True)
	stock_b.reset_index(inplace = True)

	#stock_a and stock_b contain all price information

	#next, calculate residuals
	residuals = get_residuals(stock_data, t1, t2, column='norm_close')

	

	#using boiler bands as buy signals
	bb_mean, bb_top, bb_bot = utils.boilerbands(residuals, 75, 1.5)

	#implement stop losses
	#stop loss on 1.5% down, 2.5% (hyperparam)

	#we have to check if close < stop loss from day prior, sell. 


	long_pos = ()
	short_pos = ()
	pnl = 0
	entry_points = []
	exit_points = []
	#begin iterating through the days to backtest
	for i, row in enumerate(bb_mean):

		if long_pos:
			if long_pos[0] == t2:
				long_pnl = (stock_b.loc[i]['adj_close'] - long_pos[1])
				long_returns = (stock_b.loc[i]['adj_close'] - long_pos[1]) / long_pos[1]

				short_pnl = (short_pos[1] - stock_a.loc[i]['adj_close'])
				#short_returns 

				if long_returns > 0.0225:
					pnl += long_pnl
					pnl += short_pnl
					print pnl
					long_pos = ()
					short_pos = ()
					exit_points.append(i)
				elif long_returns < -0.0225:
					pnl += long_pnl
					pnl += short_pnl
					print pnl
					long_pos = ()
					short_pos = ()
					exit_points.append(i)
			else:
				pass

		elif residuals.loc[i] < bb_bot.loc[i] and not short_pos:
			short_pos = (t1,stock_a.loc[i]['adj_close'], i)
			long_pos = (t2, stock_b.loc[i]['adj_close'], i)
			entry_points.append(i)
			print "BUY %s @ %.3f / SELL %s @ %.3f" % (t2, stock_b.loc[i]['adj_close'], t1, stock_a.loc[i]['adj_close'])
		#elif long_pos[1] 
		#if residual < bot boiler band and not short stock A
			#short stock A
			#long stock B
		#else if residual > top boiler band and not long stock A
			#short stock B
			#short stock A
		#else if long PNL < 2%, close all positions #sell signalsssss
			#wait 3 days
		#print 'BBTOP: %.3f, BBBOT: %.3f, Residual:%.3f, Stock A: %.3f, Stock B: %.3f' % (bb_top.loc[i], bb_bot.loc[i], residuals.loc[i], stock_a.loc[i]['norm_close'], stock_b.loc[i]['norm_close'])


	plt.plot(range(0, len(bb_top)), stock_a['adj_close'], label = t1)
	plt.plot(range(0, len(bb_top)), stock_b['adj_close'], label = t2)

	
	for e in exit_points:
		plt.axvline(x=e, c = 'red')
	for b in entry_points:
		plt.axvline(x=b, c = 'green')
	plt.show()


#benchmark(all_data, stock_set[0], stock_set[1])

#stop losses



