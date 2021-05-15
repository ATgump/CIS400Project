import pandas as pd
import pymongo
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report

## Fit a model on the supplied data set. Return the fit model 
def fitModel(model,model_name,X_train,y_train):
	print(model_name + " fitting...")
	model.fit(X_train,y_train)
	return model

## Test a model on the supplied data set. Return a classification report for the model
def testModel(model,model_name,X_test,y_test):
	print(model_name + " Report: \t\t")
	pred_Model = model.predict(X_test)
	report = classification_report(y_test,pred_Model)
	print(report)
	return report


if __name__ == "__main__":
	pd.set_option('display.max_rows', 500)
	
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']
	df = pd.DataFrame(list(db['processed_Training_Data'].find({},{'_id':0,'lema_text':1,'label':1})))

## CREATE AND SAVE TFIDF VECTORIZOR ### 
	vectorizor = TfidfVectorizer(min_df=.000027)
	v = vectorizor.fit_transform(df['lema_text'].to_numpy())
	
	TF_Vec = '.\\trained_Models\\TFIDF_Vectorizer.sav'
	#joblib.dump(vectorizor, TF_Vec)

	X_train,X_test,y_train,y_test = train_test_split(v,df['label'],test_size=.2,random_state=2372017502)
	
### TRAIN AND SAVE MODELS/Reports ### 

	Models = [
		('Multinomial_Naive_Bayes',MultinomialNB()),
		('Default_Multi_Layer_Perceptron',MLPClassifier()),
		('Optimized_Multi_Layer_Perceptron',MLPClassifier(hidden_layer_sizes = (1000,))),
		('Support_Vector',SVC()),
		('Decision_Tree',DecisionTreeClassifier()),
		('Gaussian_Naive_Bayes',GaussianNB()),
		('Random_Forest',RandomForestClassifier()),
		('Linear_Support_Vector',LinearSVC()),
		]

	for (model_name,model) in Models:
		path_Model ='.\\trained_Models\\'+ model_name + "_Trained_Model.sav"
		path_Report = '.\\trained_Models\\model_Classification_Reports\\'+ model_name + "_Report_Table.txt"
		

		# Data was not in proper format for Gaussian naive bayes, it is possible
		# we did not properly transform the data to train this model properly which
		# would explain the lower performance for this model compared to the others

		if model_name == 'Gaussian_Naive_Bayes':
			trained = fitModel(model,model_name,X_train.toarray(),y_train)
			report = testModel(trained,path_Report,X_test.toarray(),y_test)
		else:
			trained = fitModel(model,model_name,X_train,y_train)
			report = testModel(trained,path_Report,X_test,y_test)
		
		## Save the Model/Report ## 
		# joblib.dump(trained, path_Model)
		# with open(path_Report,'w') as file:
		#  	file.write(report)
