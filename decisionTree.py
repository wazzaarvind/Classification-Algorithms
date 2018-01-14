# Shifting logic to numpy instead of pandas

import numpy as np
import pandas as pd
import math
import time
import datetime
from collections import Counter
from sklearn.model_selection import KFold

maxDepth = 2

class node:
	# Constructor definition
	def __init__(self, initDepth, nodeParentInstance, initMaxGain, initChosenAttributeIndex, initChosenMidValue):
		
		self.nodeDepth =  initDepth
		self.parentNode = nodeParentInstance
		self.maxGain = initMaxGain
		self.chosenAttributeIndex = initChosenAttributeIndex
		self.chosenMidValue = initChosenMidValue
		
		self.leftNode = None
		self.rightNode = None
		self.classification = None

	def printNode(self, nodeObject):
		print("Node Depth: ",self.nodeDepth)
		#if(self.nodeDepth != 0):
			#print("Parent Node Data: ",nodeObject.printNode(nodeObject))
		print("left Node: ",self.leftNode)
		print("Right Node: ",self.rightNode)
		#print("ValidList: ",self.validList)
		print("Gain: ",self.maxGain)
		print("Attribute Index: ",self.chosenAttributeIndex)
		print("Mid Value: ",self.chosenMidValue)

# Read the input file
def readFile():

	allRecords = pd.DataFrame()
	#fileName = input("Enter the fileName: ")
	#fileName = "project3_dataset1.txt"
	fileName = "project3_dataset2.txt"
	
	inputFile = open(fileName,'r')

	# Reading file using Pandas 
	allRecords = pd.read_csv(fileName, sep="\t", header=None)

	listObjects = findCategoriesFields(allRecords) # Here allRecords is of type pandas

	npAllRecords = allRecords.values

	return npAllRecords, listObjects

def findCategoriesFields(allRecords): # Called from dataframeManip() and from readFile(); allRecords is of type pandas

	# Find the number of columns in a record
	noOfColumns = len(allRecords.columns)
	noOfRows = len(allRecords.index)

	listObjects = []
	for i in range(0,noOfColumns):
		if(allRecords[i].dtypes == "object"):
			listObjects.append(i)

	for i in listObjects:
		allRecords[i] = allRecords[i].astype('category')

	for i in listObjects:
		allRecords[i] = allRecords[i].cat.codes

	return listObjects

def crossValidData(npRecFromFile): # Accepts a numpy.ndarray()

	# Convert to numpy array for manipulation
	kf = KFold(n_splits=10)

	allRecords, listCatIndex = readFile() # recordsFromFile is a numpy object

	trainingDataLists = []
	testingDataLists = []

	for trainingData, testingData in kf.split(allRecords):
		trainingDataLists.append(trainingData)
		testingDataLists.append(testingData)

	return trainingDataLists, testingDataLists

# Find the indices for yes and no 
def findTotalEntropy(allRecords): # Function accepts a numpy.ndarray() and returns some integers
	onesIndex = []
	zerosIndex = []
	ones = 0
	zeros = 0

	# Find the number of columns in a record
	noOfColumns = allRecords.shape[1]
	noOfRows = allRecords.shape[0]
	
	for i in range(0,noOfRows):
		if(allRecords[i][noOfColumns-1] == 1):
			onesIndex.append(i)
			ones+=1
		else:
			zerosIndex.append(i)
			zeros+=1

	if(ones == 0 or noOfRows == 0):
		onesFormula = 0
	else:
		onesFormula = (-1 * (ones/noOfRows) * math.log((ones/noOfRows),2))

	if(zeros == 0 or noOfRows == 0):
		zerosFormula = 0
	else:
		zerosFormula = (-1 * (zeros/noOfRows) * math.log((zeros/noOfRows),2))
	
	totalEntropyFunc = onesFormula + zerosFormula

	return totalEntropyFunc, zeros, ones, zerosIndex, onesIndex

def dataframeManip(recordsToRun, listObjects): # Function accepts a numpy.ndarray() and a list with "object" attributes index

	allRecords = recordsToRun

	# Find the number of columns in a record
	noOfColumns = allRecords.shape[1]
	noOfRows = allRecords.shape[0]

	#
	### Function Called ### 
	# Move to readFile doesnt have to be performed all the time; Also make sure that the dataframe being changed is returned 
	# If only local copy is changing we will not have the updated field at all
	#

	totalEntropy, zeros, ones, zerosIndex, onesIndex = findTotalEntropy(allRecords) # Function accepts a numpy.ndarray()

	if(listObjects == []):
		return None, None, totalEntropy, None
	#
	### Function Called ###  PLEASE NOTE THAT HERE IMPLEMENTATION CHANGES 
	#

	# Count the number of unique types of variables in a series
	countUnique = dict()
	for i in listObjects:
		countUnique[i] = np.unique(allRecords[:,i])
	#print(countUnique)

	# Dictionary of dictionary of lists building data structure skeleton
	k = 0 
	indexDict = dict()
	for catIndex in listObjects:
		indexDict[catIndex] = dict()
		listOfUniqueValues = countUnique[catIndex] # Returns a list of unique values occupying the catIndex's attribute
		for j in listOfUniqueValues:
			indexDict[catIndex][j] = []
		k+=1

	# Populate the dict of lists
	# For all rows in the file, for all nominal attributes index 
	for row in range(0,noOfRows):
		for i in listObjects: 
			indexDict[i][int(allRecords[row][i])].append(row)

	return listObjects, indexDict, totalEntropy, zerosIndex

def nominalData(recordsToRun, listObjects, indexDict, totalEntropy, zerosIndex):
	#
	### Handling NOMINAL DATA ###
	#

	allRecords = recordsToRun

	# Find the number of columns in a record
	noOfColumns = allRecords.shape[1]
	noOfRows = allRecords.shape[0]

	# To calculate the entropy of all data in dataframe
	entropyDict = dict()
	gainDict = dict()
	highestGain = -1

	#totalEntropy, _, _, zerosIndex, _ = findTotalEntropy(allRecords)

	# Calculate entropy and gain for nominal data
	for catIndex in indexDict:
		k = 0
		#
		### Assume that all records with the nominal field at catIndex in the dataFrame should be checked for best gain value ###
		### Here the value that is taken as a split will basically be EQUAL TO k or NOT EQUAL TO k ###
		entropyDict[catIndex] = dict() # Defining dictionary of dictionary of list
		gainDict[catIndex] = dict()
		### Selects a list of unique values the nominal attribute can assume against a specific key; ###
		### Thus, it checks the entropy values for sunny, rainy and overcast seperately ###
		total = 0
		for uniqueValue in indexDict[catIndex]: 
			
			uniqueValueIndex = indexDict[catIndex][uniqueValue]

			allIndexesList = list(range(0,allRecords.shape[0]))
			nonKAttributeIndexes = list(set(allIndexesList) - set(uniqueValueIndex))

			#
			### for attribute index = catIndex we are assumming ONLY k will be in the left ###
			#
			zerosCount = len(set(zerosIndex).intersection(set(uniqueValueIndex))) # Zeros count intersection
			onesCount = len(uniqueValueIndex) - zerosCount # Rest is one value intersection
			
			# Calculate the entropy value of specific key
			# Handling zer0 error
			if(onesCount != 0):
				oneHalf = (-1 * ((onesCount/len(uniqueValueIndex)) * math.log(onesCount/len(uniqueValueIndex),2)))
			else:
				oneHalf = 0
			if(zerosCount != 0):
				zeroHalf = (-1 * ((zerosCount/len(uniqueValueIndex)) * math.log(zerosCount/len(uniqueValueIndex),2)))
			else:
				zeroHalf = 0

			leftEntropy = oneHalf + zeroHalf

			#
			### for attribute index = catIndex we are assumming ONLY NOT k will be in the right ###
			#
			zerosCount = len(set(zerosIndex).intersection(set(nonKAttributeIndexes))) # Zeros count intersection
			onesCount = len(nonKAttributeIndexes) - zerosCount # Rest is one value intersection
			
			# Calculate the entropy value of specific key
			# Handling zero error
			if(onesCount != 0):
				oneHalf = (-1 * ((onesCount/len(nonKAttributeIndexes)) * math.log(onesCount/len(nonKAttributeIndexes),2)))
			else:
				oneHalf = 0
			if(zerosCount != 0):
				zeroHalf = (-1 * ((zerosCount/len(nonKAttributeIndexes)) * math.log(zerosCount/len(nonKAttributeIndexes),2)))
			else:
				zeroHalf = 0

			rightEntropy = oneHalf + zeroHalf
		
			LREntropy = leftEntropy + rightEntropy

			weightedLeftEntropy = (leftEntropy * len(uniqueValueIndex)) / (allRecords.shape[0])
			weightedRightEntropy = (rightEntropy * len(nonKAttributeIndexes)) / (allRecords.shape[0])

			totalWeightedEntropy = weightedLeftEntropy + weightedRightEntropy
			
			entropyDict[catIndex][uniqueValue] =  totalWeightedEntropy # Populate the entropy dictionary with the calculated entropy
			# Calculate the gain value for sunny, rainy, windy, overcast or all unique variables that can occur in one attribute 
			gainDict[catIndex][uniqueValue] = totalEntropy - entropyDict[catIndex][uniqueValue]
			
			if(highestGain <= gainDict[catIndex][uniqueValue]):
				highestGain = gainDict[catIndex][uniqueValue]
				nomAttributeIndex = catIndex
				nomAttributeValueIndex = uniqueValue

			k += 1

	return highestGain, nomAttributeIndex, nomAttributeValueIndex

def numericalData(allRecords, ignoreIndex, totalEntropy):
	#
	### For NUMERICAL DATA ###
	#

	#allRecords = recordsToRun

	# Find the number of columns in a record
	noOfColumns = allRecords.shape[1]
	noOfRows = allRecords.shape[0]
	
	# Populate listObjects for numerical data; Will contain index of numerical series data in the table
	listObjects = []

	for i in range(0,noOfColumns-1):
		if(i not in ignoreIndex):
			listObjects.append(i)

	# Find all unique elements after eliminating the duplicates
	dictUniqueValues = dict()
	
	for i in listObjects:
		dictUniqueValues[i] = np.unique(allRecords[:,i]) # Populates unique float values existing in the series

	maxGain = -1
	chosenMidValue = -1
	chosenAttributeIndex = -1

	startLoop = time.time()
	for oneValueKey in dictUniqueValues:
		oneUniqueList = dictUniqueValues[oneValueKey]
		for oneUniqueValue in oneUniqueList:
				greaterOnes = 0
				greaterZeros = 0
				lesserZeros = 0
				lesserOnes = 0
				# Runs for all rows and it needs to find out the number 
				tempCol = allRecords[:,oneValueKey]
				tempCol = np.array(tempCol)
				
				greaterArray = allRecords[allRecords[:,oneValueKey] > oneUniqueValue]
				lesserArray = allRecords [allRecords[: , oneValueKey] <= oneUniqueValue]
				
				greaterOnes = np.array(np.where(greaterArray[:,noOfColumns-1] == 1)).shape[1]
				greaterZeros = np.array(np.where(greaterArray[:,noOfColumns-1] == 0)).shape[1]
				lesserOnes = np.array(np.where(lesserArray[:,noOfColumns-1] == 1)).shape[1]
				lesserZeros = np.array(np.where(lesserArray[:,noOfColumns-1] == 0)).shape[1]
				
				greaterTotal = greaterOnes + greaterZeros
				lesserTotal = lesserOnes + lesserZeros

				if(greaterZeros == 0):
					zerosFormula = 0
				else:	
					zerosFormula = (-1 * (greaterZeros/greaterTotal)*math.log((greaterZeros/greaterTotal),2))

				if(greaterOnes == 0):
					onesFormula = 0
				else:
					onesFormula = (-1 * (greaterOnes/greaterTotal)*math.log((greaterOnes/greaterTotal),2))

				greaterEntropy =  zerosFormula + onesFormula

				if(lesserZeros == 0):
					zerosFormula = 0
				else:	
					zerosFormula =  (-1 * (lesserZeros/lesserTotal)*math.log((lesserZeros/lesserTotal),2))
				
				if(lesserOnes == 0):
					onesFormula = 0
				else:
					onesFormula = (-1 * (lesserOnes/lesserTotal)*math.log((lesserOnes/lesserTotal),2))
				
				lesserEntropy =  zerosFormula + onesFormula

				#totalEntropy, _, _, _, _ = findTotalEntropy(allRecords)
				
				info = ((greaterEntropy * greaterTotal) + (lesserEntropy * lesserTotal)) / noOfRows

				gain = totalEntropy - info 
				
				if(maxGain <= gain): # NOTE: NOTE: Here max gain holds entropy thus we are minimising entropy
					maxGain = gain
					chosenMidValue = oneUniqueValue
					chosenAttributeIndex = oneValueKey

	return maxGain, chosenMidValue, chosenAttributeIndex

def findBestGain(recordsToRun, listCatIndex): # Function accepts a numpy.ndarray() and a list with "object" attributes index

	# Given some dataframe, it will help find out the best gain amongst nominal attributes and numerical attributes
	# Nominal Portion
	funcListObjects, funcIndexDict, totalEntropy, zerosIndex = dataframeManip(recordsToRun, listCatIndex) # Function accepts a numpy.ndarray() and a list with "object" attributes index

	if(funcListObjects != None and funcIndexDict != None):
		highestGain, nomAttributeIndex, nomAttributeValueIndex = nominalData(recordsToRun, listCatIndex, funcIndexDict, totalEntropy, zerosIndex)
	
	# Numerical Portion
	funcMaxGain, funcChosenMidValue, funcChosenAttributeIndex = numericalData(recordsToRun, listCatIndex, totalEntropy)
	
	# Chosing the better of the two
	if(funcListObjects != None and funcIndexDict != None):
		if(highestGain > funcMaxGain):
			funcMaxGain = highestGain
			funcChosenAttributeIndex = nomAttributeIndex
			funcChosenMidValue = nomAttributeValueIndex

	return funcMaxGain, funcChosenMidValue, funcChosenAttributeIndex


def classifyNode(npRecForNode): # Accepts a numpy.ndarray() and returns an integer
	
	# Find the number of columns in a record
	noOfColumns = npRecForNode.shape[1]
	noOfRows = npRecForNode.shape[0]

	classification = Counter(npRecForNode[:,noOfColumns-1]).most_common(1)

	classification = classification[0][0]
	
	#unique, counts = np.unique(npRecForNode[:,noOfColumns-1], return_counts=True)

	#classification = unique[0]
	#print(unique, counts, classification)

	return classification

def calculatePurity(npRecordSet): # Accepts a numpy.ndarray() and returns an integer

	# Find the number of columns in a record
	noOfColumns = npRecordSet.shape[1]
	#noOfRows = npRecordSet.shape[0]

	impurityStatus = 0
	uniqueClass = np.unique(npRecordSet[:,noOfColumns-1])
	if(len(uniqueClass) > 1):
		impurityStatus = 1

	return impurityStatus

def runTreeBuild(treeDepth, parentNode, listOfCats, allRecords): # allRecords is of type numpy.ndarray()
	
	global maxDepth

	treeDepth += 1

	# To begin with check if the set recieved so far is a pure set
	# if it is: then continue spliting to he next level
	# else make the node a leaf node
	
	impurityStatus = calculatePurity(allRecords) # accepts a numpy.ndarray() object and returns an integer

	#
	### PURE CASE ###
	#
	# Stop execution if this node is PURE
	if(impurityStatus == 0):
		nodeInstance = node(treeDepth, parentNode, -1, -1, -1)
		# Calssify the leaf node; pass the records in pandas format
		count = classifyNode(allRecords)
		#print(count)
		nodeInstance.classification = count
		return nodeInstance

	#
	### IMPURE CASE ###
	#
	# Continue execution of the node is IMPURE
	else:
		
		# Calculate the split point; Function accepts a numpy.ndarray() and a list with "object" attributes index
		#gainTime = time.time()
		midMaxGain, midChosenMidValue, midChosenAttributeIndex = findBestGain(allRecords, listCatIndex) 
		#print("Gain time: ",time.time() - gainTime)

		### Check if the field is a continous attribute or a nominal attribute ###

		#
		### NOMMINAL ATTRIBUTE ###
		#
		if(midChosenAttributeIndex in listOfCats):
			# It is a nominal attribute
			# Populate the list for left and right valid index
			# Convert to numpy for manipulation
			npAllRecords = allRecords

			# Left sub set should have only equality satisfying values
			tempLeftSubRec = npAllRecords[npAllRecords[:,midChosenAttributeIndex] == midChosenMidValue]

			# Right sub set should have only non-equality satisfying values
			tempRightSubRec = npAllRecords[npAllRecords[:,midChosenAttributeIndex] != midChosenMidValue]

			if(tempRightSubRec.shape[0] > 0):
				
				#
				### Make current node a NON-LEAF NODE ###
				#
				# Create a current node instance and call for filling out its left and right nodes
				# Now call the runTreeBuild() recursively since the method can now be further split
				currentNode = node(treeDepth, parentNode, midMaxGain, midChosenAttributeIndex, midChosenMidValue)
				
				# This recursive call will return a left chid node here
				currentNode.leftNode = runTreeBuild(treeDepth, currentNode, listOfCats, tempLeftSubRec)
				
				# This recursive call will return a right chid node here
				currentNode.rightNode = runTreeBuild(treeDepth, currentNode, listOfCats, tempRightSubRec)
			
			else:
				# Assign a new left block node with some details and end here by returning the parent
				currentNode = node(treeDepth, parentNode, -1, -1, -1)
				count = classifyNode(allRecords)
				currentNode.classification = count



		#
		### NUMERICAL ATTRIBUTE ###
		#
		else:
			# It is a numerical attribute
			# Populate the list for left and right valid index
			# Convert to numpy for manipulation
			npAllRecords = allRecords
			
			#first = time.time()
			tempLeftSubRec = npAllRecords[npAllRecords[:,midChosenAttributeIndex] < midChosenMidValue]
			tempRightSubRec = npAllRecords[npAllRecords[:,midChosenAttributeIndex] >= midChosenMidValue]
			#firstTime = time.time() - first

			# If the left is not empty continue execution
			if(tempLeftSubRec.shape[0] > 0):
				#
				### Make current node a NON-LEAF NODE ###
				#
				# Now call the runTreeBuild() recursively since the method can now be further split
				
				# Create a current node instance to populate children to
				currentNode = node(treeDepth, parentNode, midMaxGain, midChosenAttributeIndex, midChosenMidValue)
				
				# This recursive call will return a left chid node here
				currentNode.leftNode = runTreeBuild(treeDepth, currentNode, listOfCats, tempLeftSubRec)
				
				# This recursive call will return a right chid node here
				currentNode.rightNode = runTreeBuild(treeDepth, currentNode, listOfCats, tempRightSubRec)
			
			else:
				# Assign a new left block node with some details and end here by returning the parent
				currentNode = node(treeDepth, parentNode, -1, -1, -1)
				count = classifyNode(allRecords)
				currentNode.classification = count

	return currentNode

def testingModule(oneTestRecord, node, listOfCats):

	if(node.leftNode == None):
		return node.classification

	else:
		# Now find if it is Numerical or Nominal
		if(node.chosenAttributeIndex in listOfCats):
			# Handle for nominal data

			if(node.chosenMidValue == oneTestRecord[node.chosenAttributeIndex]):
				classification = testingModule(oneTestRecord, node.leftNode, listOfCats)

			else:
				classification = testingModule(oneTestRecord, node.rightNode, listOfCats)
		else:
			# Handle for numerical data

			if(oneTestRecord[node.chosenAttributeIndex] < node.chosenMidValue):
				classification = testingModule(oneTestRecord, node.leftNode, listOfCats)

			else:
				classification = testingModule(oneTestRecord, node.rightNode, listOfCats)

		return classification

def calculateAccuracy(predictedValue, testRecords):

	similarity = 0

	for i in range(0,len(predictedValue)):
		if(predictedValue[i] == testRecords[i][testRecords.shape[1]-1]):
			similarity += 1

	accuracyValue = (similarity/len(predictedValue)) * 100

	return accuracyValue

#						    #
##						   ##
###                       ###
### Execution starts here ###
###             		  ###
##                         ##
#							#

# Reads the whole input file

# Read Input from the file and process in pandas, returns a numpy object
recordsFromFile, listCatIndex = readFile() # recordsFromFile is a numpy object
#print("readFile: ",time.time() - startTime)

# Split the dataset into two for cross validation
trainRecordsList, testRecordsList = crossValidData(recordsFromFile) # Accepts a numpy array and returns two numpy arrays


timesRun = 0
accuracyValue = 0

accuracyValue = []
precision = []
recall = []
f_measure = []

for i in range(0, len(trainRecordsList)): # Run for that many sets of training data
	
	npTrainRecords = recordsFromFile[trainRecordsList[i]]

	rootNode = runTreeBuild(-1, None, listCatIndex, npTrainRecords) # Build the decision tree based on a list of training data
	
	resultsList = []

	a=0
	b=0
	c=0
	d=0

	for index in testRecordsList[i]: # Extracting an index from a list of training data indexes 
		
		npTestRecordsListRow = recordsFromFile[index]
		resultsList.append(testingModule(npTestRecordsListRow, rootNode, listCatIndex))
	
	#npTestRecList = recordsFromFile[testRecordsList[i]]

	groundTruth = recordsFromFile[testRecordsList[i] , recordsFromFile.shape[1]-1]
	
	for y in range(0,len(groundTruth)):
		if(groundTruth[y]==1 and resultsList[y]==1):
			a+=1
		elif(groundTruth[y]==1 and resultsList[y]==0):
			b+=1
		elif(groundTruth[y]==0 and resultsList[y]==1):
			c+=1
		elif(groundTruth[y]==0 and resultsList[y]==0):
			d+=1

	print("Metrics for fold iteration: ", i+1)
	print("Accuracy: ",((a+d)/(a+b+c+d)))
	print("Precision: ",(a/(a+c)))
	print("Recall: ",(a/(a+b)))
	print("F Measure: ",((2*a) / ((2*a)+b+c)))
	print("\n")

	accuracyValue.append((a+d)/(a+b+c+d))
	precision.append(a/(a+c))
	recall.append(a/(a+b))
	f_measure.append((2*a) / ((2*a)+b+c))

print("Average score for all",len(trainRecordsList)," records:")
print("Accuracy is: ",np.mean(accuracyValue))
print("Precision is: ",np.mean(precision))
print("Recall is: ",np.mean(recall))
print("F-Measure is: ",np.mean(f_measure))



