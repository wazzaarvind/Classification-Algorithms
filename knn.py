
import numpy as np
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from scipy import stats


categorical_index =[]

def readFile(fileName):
	file=open(fileName , 'r')
	line = file.readline()
	columnsLimit = len(line.strip().split('\t'))


	feature = line.strip().split("\t")
	for k in feature:

 		if k.isalpha():
 			categorical_index.append(feature.index(k))




	matrix = np.genfromtxt(fileName , delimiter = '\t' , dtype=None  )

	matrix=np.array(matrix.tolist())


	for i in categorical_index:
		labelencoder=LabelEncoder()
		matrix[:,i] = labelencoder.fit_transform(matrix[:,i])

	matrix = np.array(matrix).astype(float)

	class_labels=np.array(matrix[:,-1:])
	class_labels = np.array(class_labels)
	matrix = matrix [:,:-1]

	return matrix , class_labels




def call_Kfold(mat , kvalue ):
	k=int(kvalue)
	
	

	kf = KFold(n_splits=10)
	kf.get_n_splits(mat)

	#k=int(kvalue)
	accuracy=[]
	precision=[]
	recall=[]
	f_measure=[]

	categ_mul=[]
	

	for train_index , test_index in kf.split(mat):
		actual_class=[]
		predicted_class=[]
		a=0
		b=0  
		c=0 
		d=0

		result=0
		distance_matrix=np.zeros( (len(test_index) , len(train_index)) , dtype=float)


		for i in range(0,len(test_index)):
			#print('Test index :' , test_index[i])
			actual_class.append(class_labels[test_index[i]])
			#print('Its class label ' , class_labels[test_index[i]])
			for j in range(0,len(train_index)):
				
				distance_matrix[i][j] = np.linalg.norm(mat[test_index[i]] - mat[train_index[j]])



		actual_class=np.array(actual_class)

		for x in range(0,len(distance_matrix)):
			count_0=0
			count_1=0
			temp=[]
			#print('Distance matrix is ' , distance_matrix[x])
			mini=0
			mini=np.argsort(distance_matrix[x])[:k]
			#print(mini)
		

			for y in mini:
				temp.append(class_labels[train_index[y]])

			for l in range(0,len(temp)):
				if(temp[l]==0):
					count_0+=1
				else:
					count_1+=1
			
			#print('Count 0 is ' , count_0 , 'Count 1 is ' , count_1)
			if(count_0<count_1):
				predicted_class.append(1)
			else:
				predicted_class.append(0)


		


		for o in range(0,len(actual_class)):
			if(actual_class[o]==1 and predicted_class[o]==1):
				a+=1
			elif(actual_class[o]==1 and predicted_class[o]==0):
				b+=1
			elif(actual_class[o]==0 and predicted_class[o]==1):
				c+=1
			elif(actual_class[o]==0 and predicted_class[o]==0):
				d+=1
		
		accuracy.append((a+d)/(a+b+c+d))
		precision.append((a)/(a+c))
		recall.append(a/(a+b))
		f_measure.append((2*a)/((2*a)+b+c))




	print('Accuracy: ' , np.mean(accuracy))

	print('Precision: ' , np.mean(precision))
	print('Recall: ' , np.mean(recall))
	print('F-Measure: ' , np.mean(f_measure))
	

		#print('F measure last' , ( (2*np.mean(recall) * np.mean(precision)) / (np.mean(recall) + np.mean(precision)) ) )

#input_file = "project3_dataset2.txt"

input_file = input('Enter the name of the input file: ')
k = input('Enter value of K : ')
matrix , class_labels = readFile(input_file)



#min_max_scaler = preprocessing.MinMaxScaler()
#matrix_minmax = min_max_scaler.fit_transform(matrix)


matrix_minmax = stats.zscore(matrix)


call_Kfold(matrix_minmax , k )

