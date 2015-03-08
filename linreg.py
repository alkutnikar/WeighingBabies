import pandas as pd
import numpy as np
import matplotlib.pyplot as plt







babiesDF = pd.read_csv('./2008_births.csv')
slicedDF = pd.DataFrame
trainDF = pd.DataFrame
testDF = pd.DataFrame
evalDF = pd.DataFrame

colIndex = 0
def binarizeColumn(x):
    global colIndex
    if x == colIndex:
        return 1
    else:
        return 0

def binarizeHisp(x):
    if x == 'N':
        return 0
    else:
        return 1

#global colIndex
#global slicedDF
requiredColumns = ['SEX', 'MARITAL','FAGE', 'MAGE', 'FEDUC', 'MEDUC', 'TOTALP', 'BDEAD', 'TERMS', 'LOUTCOME', 'WEEKS', 'BPOUND', 'BOUNCE', 'WEIGHT', 'RACEMOM', 'RACEDAD', 'HISPMOM', 'HISPDAD', 'CIGNUM', 'DRINKNUM', 'GAINED', 'ANEMIA', 'CARDIAC', 'ACLUNG', 'DIABETES', 'HERPES', 'HYDRAM', 'HEMOGLOB', 'HYPERCH', 'HYPERPR', 'ECLAMP',
                             'CERVIX','PINFANT','PRETERM','RENAL','RHSEN','UTERINE']
slicedDF = babiesDF.ix[:,requiredColumns]

print "Number of rows before removal are", len(slicedDF)

#Removing rows with missing weight details
slicedDF = slicedDF[slicedDF['BPOUND']!=99]
slicedDF = slicedDF[slicedDF['BOUNCE']!=99]
len(slicedDF)

#adding a new column for weight in pounds
weightPound = slicedDF['BPOUND']
weightOunces = slicedDF['BOUNCE']
weight = weightPound.astype(np.float) + (0.0625 * weightOunces.astype(np.float))
slicedDF['WEIGHTLB'] = weight
raceColumns = ['OTHER_NON_WHITE', 'WHITE', 'BLACK', 'AMERICAN_INDIAN', 'CHINESE', 'JAPANESE', 'HAWAIIAN', 'FILIPINO', 'OTHER_ASIAN']
for race in raceColumns:
    slicedDF[race + '_MOM'] = slicedDF['RACEMOM'].map(binarizeColumn)
    colIndex = colIndex + 1
colIndex = 0
for race in raceColumns:
    slicedDF[race + '_DAD'] = slicedDF['RACEMOM'].map(binarizeColumn)
    colIndex = colIndex + 1
slicedDF['HISPMOM_BINARY'] = slicedDF['HISPMOM'].map(binarizeHisp)
slicedDF['HISPDAD_BINARY'] = slicedDF['HISPDAD'].map(binarizeHisp)
columnsToDrop = ['BPOUND','BOUNCE','RACEMOM','RACEDAD','HISPMOM','HISPDAD']
for column in columnsToDrop:
    slicedDF = slicedDF.drop(column, 1)

missing99Columns = ['MAGE','FEDUC','MEDUC','TOTALP','BDEAD','TERMS','WEEKS','WEIGHT','CIGNUM','DRINKNUM','GAINED']
missing9Columns = ['LOUTCOME','ANEMIA','CARDIAC','ACLUNG','DIABETES','HERPES','HYDRAM','HEMOGLOB','HYPERCH','HYPERPR','ECLAMP'
                   ,'CERVIX','PINFANT','PRETERM','RENAL','RHSEN','UTERINE']
missingValDF = slicedDF[slicedDF['FAGE']!=99]

for col in missing99Columns:
    missingValDF = missingValDF[missingValDF[col]!=99]
    
for col in missing9Columns:
    missingValDF = missingValDF[missingValDF[col]!=9]

print "Number of rows after removing missing values:",len(missingValDF)

def get_splitDF():
        global missingValDF
        global trainDF
        global testDF
        global evalDF
        DF_temp = pd.DataFrame
        rand_nos = np.random.rand(len(missingValDF)) < 0.7
        trainDF = missingValDF[rand_nos]
        DF_temp = missingValDF[~rand_nos]

        rand_nos = np.random.rand(len(DF_temp)) < 0.6
        testDF = DF_temp[rand_nos]
        evalDF = DF_temp[~rand_nos]

        print 'Train(len) : {0} rows'.format(str(len(trainDF)))
        print 'Test(len) : {0} rows'.format(str(len(testDF)))
        print 'Eval(len) : {0} rows'.format(str(len(evalDF)))

get_splitDF()

trainWeights = trainDF['WEIGHTLB']
testWeights = testDF['WEIGHTLB']
evalWeights = evalDF['WEIGHTLB']
trainDF = trainDF.drop(['WEIGHTLB','WEIGHT'], 1)
testDF = testDF.drop(['WEIGHTLB','WEIGHT'], 1)
evalDF = evalDF.drop(['WEIGHTLB','WEIGHT'], 1)

#Converting our final dataframe into a n-dimensional list. 
#This is required for passing our data to any interpolation model.
#print trainDF.columns[30]
allRowsList = []
for idx,row in trainDF.iterrows():
    currentRowList = row.values.tolist()
    allRowsList.append(currentRowList)

def runRegression(type, passedDf, passedWeights):
	newTestDF = pd.DataFrame()
	newTestDF = passedDf.copy()
	from sklearn import linear_model
	if type == 0:
		clf = linear_model.LinearRegression()
	elif type == 1:
		clf = linear_model.Ridge(alpha=0.5)
	clf.fit(allRowsList, trainWeights)
	coeffArray = clf.coef_
	intercept = clf.intercept_ 
	print "Co-efficients of linear regression are", coeffArray
	print "Intercept is",intercept

	testWeightsList = passedWeights.tolist()
	i = 0
	olsPrediction = []
	for idx,row in newTestDF.iterrows():
	    currentRowList = row.values.tolist()
	    predictedWeightRegression = np.dot(coeffArray,currentRowList) + intercept
	    olsPrediction.append(predictedWeightRegression)
	    #print "Predicted Weight is", predictedWeightRegression
	    actualWeight = testWeightsList[i]
	    #print "Actual Weight is", actualWeight
	    i = i + 1

	sammyRow = [2, 1, 27, 27, 12, 16, 1, 0, 0, 1, 40, 0, 0.25, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]

	if len(passedDf) < 10000:
		if type == 0:
			print "Prediction for Sammy's child according to OLS Regression is", np.dot(coeffArray,sammyRow) + intercept
		else:
			print "Prediction for Sammy's child according to Ridge Regression is", np.dot(coeffArray,sammyRow) + intercept
	if type == 0:
		newTestDF['PREDICTED_WEIGHT_OLS'] = olsPrediction
	else:
		newTestDF['PREDICTED_WEIGHT_RIDGE'] = olsPrediction
	newTestDF['ACTUAL_WEIGHT'] = passedWeights
	newTestDF['BASELINE_WEIGHT'] = np.mean(testWeightsList)

	import math
	newTestDF['BASELINE_SQUARED_ERROR'] = (newTestDF['BASELINE_WEIGHT'] - newTestDF['ACTUAL_WEIGHT']) * (newTestDF['BASELINE_WEIGHT'] - newTestDF['ACTUAL_WEIGHT'])
	baseLineRMS = math.sqrt(np.mean(newTestDF['BASELINE_SQUARED_ERROR']))
	print "Root Mean Squared Error for baseline model is", baseLineRMS

	if type == 0:
		newTestDF['OLS_SQUARED_ERROR'] = (newTestDF['PREDICTED_WEIGHT_OLS'] - newTestDF['ACTUAL_WEIGHT']) * (newTestDF['PREDICTED_WEIGHT_OLS'] - newTestDF['ACTUAL_WEIGHT'])
		olsRMS = math.sqrt(np.mean(newTestDF['OLS_SQUARED_ERROR']))
		print "Root Mean Squared Error for OLS model is", olsRMS
	else:
		newTestDF['RIDGE_SQUARED_ERROR'] = (newTestDF['PREDICTED_WEIGHT_RIDGE'] - newTestDF['ACTUAL_WEIGHT']) * (newTestDF['PREDICTED_WEIGHT_RIDGE'] - newTestDF['ACTUAL_WEIGHT'])
		olsRMS = math.sqrt(np.mean(newTestDF['RIDGE_SQUARED_ERROR']))
		print "Root Mean Squared Error for Ridge Regression model is", olsRMS

	import seaborn as sns
	sns.set_palette("hls")
	#mpl.rc("figure", figsize=(8, 4))
	ax1 = plt.subplot(211)
	ax1 = sns.distplot((newTestDF['BASELINE_WEIGHT'] - newTestDF['ACTUAL_WEIGHT']), bins = 20)
	ax1.set_xlim((-8,8))
	ax1.set_title('Baseline Model')
	ax2 = plt.subplot(212)
	if type == 0:
		ax2 = sns.distplot((newTestDF['PREDICTED_WEIGHT_OLS'] - newTestDF['ACTUAL_WEIGHT']), bins = 20)
		ax2.set_title('OLS Model')
	else:
		ax2 = sns.distplot((newTestDF['PREDICTED_WEIGHT_RIDGE'] - newTestDF['ACTUAL_WEIGHT']), bins = 20)
		ax2.set_title('Ridge Regression Model')
	ax2.set_xlim((-8,8))
	
	#plt.title('Probability Distribution of Error')

	plt.show()
	

print "-----------------Running OLS Regression-------------------"
print "--------------Test Data------------------"
runRegression(0,testDF, testWeights)
print "--------------Evaluation Data------------------"
runRegression(0,evalDF, evalWeights)
print "-----------------Running Ridge Regression-------------------"
print "--------------Test Data------------------"
runRegression(1,testDF, testWeights)
print "--------------Evaluation Data------------------"
runRegression(1,evalDF, evalWeights)




