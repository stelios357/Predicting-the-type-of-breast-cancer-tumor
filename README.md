# Predicting-the-type-of-breast-cancer-tumorIntroduction to Data Science


Department of Computer Science Engineering
The LNM Institute of Information Technology, Jaipur






TABLE OF CONTENTS



1.	Project Objective. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
2.	Dataset Description. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
3.	Data Set and Library imports . . . . . . . . . . . . . . . . . . . . . . . 5
4.	Data Analysis and Preprocessing . . . . . . . . . . . . . . . . . . . 7
5.	Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
6.	ML classification . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
7.	Result . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
8.	References . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26














PROJECT OBJECTIVE 

The goal of this project is to train classification models to predict the type of breast cancer tumor (Malignant and Benign) and use this learned model to predict the tumor type when given its features. A brief comparison of different classification models used is also done.

DATASET DESCRIPTION

Data Set: Breast Cancer Wisconsin (Diagnostic) dataset 
Number of Instances: 569
Number of Attributes:32
Area: Health & Medical Sciences
Task: Classification

The features of the dataset are evaluated from a digitized image of a Fine Needle Aspirate (FNA) of a breast mass. They represent the characteristics of the cell nuclei in the image.
Attribute Information:
1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)
Ten real-valued features are computed for each cell nucleus:
a) radius (mean of distances from the center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)

The mean, standard error, and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.
All feature values are recorded with four significant digits.

SOURCE: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
DATASET AND LIBRARY IMPORTS

 
Uploading the dataset to the google collab server from our system.


 

Importing all the necessary libraries required by the program.
●	The NumPy library is used for working with arrays. It has many functions that are useful in the field of linear algebra, matrices, and Fourier transformation.
●	The pandas library is a useful tool to deal with data that is stored in tabular form. It offers data structures and functions for manipulating and analyzing numerical data.
●	The seaborn and matplotlib libraries are powerful data visualization libraries with many functions to help visualize our data using different plots...
●	Scipy is a Python library that uses NumPy for more mathematical operations. 
●	Sklearn is a very powerful Python library used in Machine Learning. It provides various classification algorithms. It also supports libraries like numpy and scipy.


df=pd.read_csv("data.csv")

The .read_csv() method read the comma-separated file into a DataFrame named df.



DATA ANALYSIS AND PRE- PROCESSING

Data Analysis 

df.head(10)


The .head() method displays the top 5 (we have used the parameter 10 for our convenience) instances in the dataset which gives us quick information about whether the object has the right type of data in it. One observation we get from the output of this method is that the diagnosis attribute is the class label.
 

df.info()
 
The .info() method allows us to learn the shape of our data. It provides us with the number of instances in the dataframe, number of attributes, attribute names , attribute data types, attribute non-null count. 
The following observations are made:

●	The data has 33 attributes, out of which two attributes come to attention, id and Unnamed: 32. The attribute id is of no use for classification as it is a key attribute (unique for each record) and Unnamed: 32 includes NaN values, so we do not require it. 
●	There is no null value in the attributes except Unnamed: 32 for any instance in the dataset. Hence, there is no need to handle null values.
●	The RangeIndex varies from 0 to 568, suggesting that the data has 569 rows.
●	Except diagnosis, all the attributes are numeric in nature.


 df.describe()

 

The .describe() method gives summary statistics for the numerical attributes in the dataframe df. We notice that the range of values for attributes vary greatly; therefore we normalize the data to minimize the training model sensitivity to the scale of attributes.






df.columns
 
This gives us the attribute names of the dataframe df as a Python list.

Data Pre- Processing

In Analysis, we established the need for normalization and  non essential nature of the attributes id and Unnamed: 32. So, in pre-processing, we drop the non essential attributes and normalize the data in the dataframe df.

 

The variable y stores the diagnosis attribute values as a Pandas Series which is later used in the classification model.
The dropped_df data frame has dropped the attributes id,Unnamed: 32 and diagnosis .This dataframe is later used for classification. The class label attribute is removed because a portion of the dataframe is later used as testing data for which the class label cannot be predefined.
We remove the non essential attributes id, Unnamed: 32 from the dataframe df and use the method .head() to ensure the attributes are dropped off.
 


 

The normalization is done for the data frame dropped_df which is used for the training and testing of the classification model to minimize the sensitivity of the model to the scale of attributes. The .head() method is used to ensure that the dataframe is normalized. The .MinMaxScalar() estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.
 


 
 

Once the data is normalized, non essential attributes are dropped and stored in the dataframe drop_norm_df, the maximum and minimum values for each attribute is printed for each class label, Malignant (M) and Benign (B).
 
 
DATA VISUALIZATION

1.	Count Plot

 

B:  357                   M: 212
 
●	The count plot gives us a graphical representation of the number of instances for each class label, Malignant (M),represented by blue and Benign (B),represented by orange using the Pandas Series y.
●	The number of records in the dataframe belonging to class Malignant (M) is 357 and class Benign (B) is 212.
●	The importance of count plots is to check if classes are imbalanced, which produces poor insights. From the above plot, we observe that the classes are not greatly imbalanced.
2.	Pair Plot

 

This produces a pairplot of the dataframe df for the attributes in the list cols.
This plot helps us to visualize the relationship between each pair of attributes in the list cols for each class label. The Malignant (M) class is represented by blue and Benign (B) class is represented by orange. Pair plot gives a matrix of relationships between each attribute for an instant examination of our data. 
 
3.	Heatmap

 

 

It produces a heatmap for the dataframe df. This plot visualizes the correlation between each pair of attributes in the dataframe df. This is done as some classification models (like decision tree based) have the prerequisite that attributes be not correlated to each other.

4.	Box Plot

 

 

It produces box plots for each attribute of the normalized data frame drop_norm_df. The Malignant (M) class is represented by blue and the Benign (B) class is represented by orange. Box plots are very useful to see outliers in the data. Some classification models are sensitive to outliers, hence the need to check for outliers in the data.
ML CLASSIFICATION

The project uses two ML models or algorithms for prediction of the class labels of testing dataset i.e (x_test). Two models are used in order to compare their working and the results produced by them.

 

The output of the above code:

This creates a 569 ✕ 2 matrix with values 0 or 1. Column 1 takes value as 0 if the diagnosis is negative for Benign(B) and 1 if it is positive for Benign(B). Similarly, column 2 takes value as 0 if the diagnosis is negative for Malignant(M) and 1 if it is positive for Malignant(M).
The dataset is divided into two, x_train (for training of the model) and x_test( for testing), in the ratio of 4:1. Randomization function of train test split is also used to ensure proper split of data without any bias. 
The new set of data obtained through the matrix created above is also divided into 2, y_train and y_test for training and testing. 
 

The output of the above code:
 

Once training is done, we test our model using the x_test dataset that does not have its label predefined .Similar test is performed from the y_test dataset. The predictions obtained from both the data sets are compared and based on how many of our predictions match with y_test, we get the accuracy by the respective model.

The models used in the project are:

1.	 Random Forest Classifier
	
Random Forest is a supervised learning algorithm. The basic idea behind this model is that it combines learning models to increase the overall result, in other words it builds several decision trees and then combines them to produce a more accurate prediction. 

Code for Random Forest :

 

The RandomForest Classifier is used to create a forest and the .fit function fits the x_train and y_train dataset into the forest.

The output of the above code:

 

 
The .predict function is used to make predictions for the x_test dataset using the created forest .

 

Further the accuracy , Root mean square error and classification report is generated for the Random Forest model using the x_test and y_test dataset.

The output of the above code:

 


2.	 Support Vector Machine

Support Vector Machine or SVM is a ML algorithm that is used for classification and regression. The SVM calculation makes a choice limit that can isolate n-dimensional space into classes so we can  put the new information point in the right classification later on. This best decision boundary is known as a hyperplane.

In other words, SVM is a classifier that searches for a hyperplane with largest margin, which is why it is often known as maximum margin classifier. 







The function print_score ,prints the output and confusion matrix for SVM:

 

Code for SVM:

 

For the SVM model , the training and testing sets are formed in the ratio 4:1 and a SVM model is created using LinearSVM . The .fit  fits the x_train and y_train into the model or in other words trains the model . Further the print_score method is called to print the results of SVM for the testing datasets.

The output of the above code: 

 











RESULT


Random Forest:

Accuracy = 97.37%
Root Mean Square Error = 0.162
F score =0.973
Recall=0.97
Precision =0.97
Support=114

Support Vector Machine or SVM :

For training set:
Accuracy = 97.58%
Confusion matrix = [[ 284  1][10  160]]
F score =0.976
Recall=0.979
Precision =0.976
Support=455


For testing set:
Accuracy = 98.25%
Confusion matrix = [[72  0][2  40]]
F score =0.982
Recall=0.982
Precision =0.983
Support=114

 
(Confusion matrix, F score, precision and recall are used to determine the final result instead of percent accuracy to avoid ambiguity due to class imbalance.)

Comparing the results of the two models we can say that Svm gives a more accurate result as compared to Random Forest. 

 













REFERENCES

1.	https://www.javatpoint.com/ - for definition and use of Svm and Random forest
2.	www.tutorialspoint.com - for definition and use of Svm and Random forest
3.	https://stackoverflow.com/


GOOGLE COLAB LINK

https://colab.research.google.com/drive/1Rht63tExMnsqwQ6liyOmtGpCowr3f5LS?usp=sharing

DATASET 

https://drive.google.com/file/d/14pjyPatWtm9-wvdaCXwkw4uC2yFjzPTi/view?usp=sharing
