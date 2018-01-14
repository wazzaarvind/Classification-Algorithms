import numpy as np
from sklearn.model_selection import KFold
import scipy as sc

categorical_index = []     # finding out the vindex which has categorical values
feature=[]				 # just for splitting a line to check categorical feature

categorical_map=dict()



def readfile(fileName):

	categorical_features=[]

	file = open(fileName , 'r')
	line = file.readline()

	columnsLimit = len(line.strip().split("\t"))
	column_names = [str(x) for x in range(0,columnsLimit)]

	
	feature = line.strip().split("\t")
	for k in feature:

 		if k.isalpha():
 			categorical_index.append(feature.index(k))

		
	
	matrix = np.genfromtxt(fileName , delimiter = '\t' , dtype=None , names=column_names )

	class_labels=np.array(matrix[str(columnsLimit-1)])         
	
	matrix=np.array(matrix.tolist())


	#matrix = matrix[:,:-1]

	

	for i in categorical_index:
		#categorical_features.append([])
		categorical_features.append(matrix[:,i])


	matrix_temp = np.delete( matrix, categorical_index , axis=1)				#deleting categorical features from this matrix. and storing it in a separate matrix
	matrix_temp = np.array(matrix_temp).astype(np.float)

	matrix_temp = matrix_temp[:,:-1]

	categorical_features=np.array(categorical_features)

	return matrix , matrix_temp, class_labels ,categorical_features , columnsLimit   # original matrix with categorical features ,feature matrix without categorical and class labels





def calculate_class_prior_probability(train_index):
	number_of_1 = 0
	number_of_0 = 0

	indices_0=[]
	indices_1=[]


	for k in train_index:
		if class_labels[k]==0:
			number_of_0+=1
			indices_0.append(k)
		else:
			number_of_1+=1
			indices_1.append(k)


	
	class_probability_0 = (number_of_0 / len(train_index))				 
	class_probability_1 = (number_of_1 / len(train_index))

	return class_probability_0 , class_probability_1 , number_of_0 , number_of_1 , indices_0 , indices_1



def calculate_posterior_probability(train_index , test_index , number_of_0 , number_of_1):



	categorical_map.clear()
	matrix_0=[None]*number_of_0
	matrix_1=[None]*number_of_1
	c0=0
	c1=0
	cf0=[]
	cf1=[]

	class_label0=[]
	class_label1=[]

	for k in train_index:
		if class_labels[k]==0:
			matrix_0[c0]=matrix_temp[k]      # contains rows from the matrix_temp whose class labels are 0
			class_label0.append(0)
			c0+=1
		else:
			matrix_1[c1]=matrix_temp[k]	    # contains rows from the matrix_temp whose class labels are 1
			class_label1.append(1)
			c1+=1



	class_label0=np.array(class_label0)
	class_label1=np.array(class_label1)
	mean_0 = np.mean(matrix_0 , axis=0)
	mean_1 = np.mean(matrix_1 , axis = 0)

	std_0 = np.std(matrix_0 , axis=0)
	std_1 = np.std(matrix_1 , axis = 0)




	
	unique_values=[]

	if categorical_features.size>0:
		nd=np.unique(categorical_features ,axis=1)


				
				



		

	



	return matrix_0 , matrix_1 , mean_0 , mean_1 , std_0 , std_1 
	






def check_categorical(test_data , j ,indices_0 , indices_1) :
	c_feature = matrix[:,j]
	v0=0
	v1=0
	#print (test_data)
	#print(indices_0)
	#print(indices_1)
	for c0 in indices_0:
		if(c_feature[c0]==test_data):
			v0+=1

	for c1 in indices_1:
		if(c_feature[c1] ==test_data):
			v1+=1
	#print(v0,v1)
	return v0,v1
	



def call_Kfold(matrix):
	kf = KFold(n_splits=10)
	kf.get_n_splits(matrix)

	accuracy=[]
	precision=[]
	recall=[]
	f_measure=[]

	categ_mul=[]

	for train_index , test_index in kf.split(matrix):
		actual_class=[]
		predicted_class=[]
		a=0
		b=0  
		c=0 
		d=0
		class_probability_0 , class_probability_1 , number_of_0 , number_of_1 , indices_0 , indices_1 = calculate_class_prior_probability(train_index)
		matrix_0 , matrix_1 , mean_0 , mean_1 , std_0 , std_1 = calculate_posterior_probability(train_index ,test_index , number_of_0 , number_of_1)



		for i in test_index:
			
			test_data = matrix_temp[i]
			#print('Test data is ' , test_data)

			actual_class.append(class_labels[i])
			#print(actual_class)



			p0=sc.stats.norm(mean_0, std_0).pdf(test_data)
			p1=sc.stats.norm(mean_1, std_1).pdf(test_data)
			p0=np.prod(p0)
			p1=np.prod(p1)				#have to multiply prior.
			#print(p0,p1)
			#test_data=matrix[i]

			if categorical_index:

				for j in categorical_index:
					test_data=matrix[i][j]
					#print('Categorcial Feature' , test_data)



					v0 , v1=check_categorical(test_data , j ,indices_0 , indices_1)
					#print(v0,v1)


					v0 = v0/number_of_0
					v1 = v1/number_of_1

					p0=p0*v0
					p1=p1*v1


						
			
			p0=p0*class_probability_0
			p1=p1*class_probability_1


			#print(p0,p1)

			if p0<p1:
				predicted_class.append(1)
			else:
				predicted_class.append(0)
		#print(predicted_class)
		for i in range(0,len(actual_class)):
			if(actual_class[i]==1 and predicted_class[i]==1):
				a+=1
			elif(actual_class[i]==1 and predicted_class[i]==0):
				b+=1
			elif(actual_class[i]==0 and predicted_class[i]==1):
				c+=1
			elif(actual_class[i]==0 and predicted_class[i]==0):
				d+=1

			
		
		accuracy.append((a+d)/(a+b+c+d))
		precision.append((a)/(a+c))
		recall.append(a/(a+b))
		f_measure.append((2*a)/((2*a)+b+c))
		

	print('Accuracy: ' , np.mean(accuracy))
	print('Precision: ' , np.mean(precision))
	print('Recall: ' , np.mean(recall))
	print('F-Measure: ' , np.mean(f_measure))
	

		


input_file = input('Enter the name of the input file : ')
matrix , matrix_temp , class_labels ,categorical_features , columnsLimit= readfile(input_file)     # gets the feature matrix and class labels
#print(categorical_features.shape)
call_Kfold(matrix)



