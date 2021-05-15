import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

## Get a random sample from a mongo collection
def sampleDB(collection,sample_size):
	return collection.aggregate([{'$sample':{'size':sample_size}}])

## Generate a pie chart for some model(s)/restaurant
def generatePieChart(values,mlabels,restaurant,model):
	plt.pie(values,labels=mlabels)
	fig = plt.gcf()
	s = restaurant + '_Pie_Chart_' + model
	plt.show()
	return fig

if __name__ == "__main__":

	## Connection to Mongo tweet_DB
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']


	# 1. Create Pie charts for different Experiments (models/restaurants)

	restaurants = ['Wendys','BK','McD']
	models = ['MLP','RF','LSV','Combined']

	for model in models:
		table = '' ## table to record the values of the pie chart 
		for restaurant in restaurants:
			table = table + restaurant + '\n'
			col = 'exp_Data_' + restaurant + '_' + model ## mongo collection to access
			if model == 'Combined':
				sample = sampleDB(db[col],2000) ## combining methods resulted in smalled data set so smaller sample taken
			else:
				sample = sampleDB(db[col],2500)	
			df = pd.DataFrame(sample)
			mlabels = ['negative','neutral','positive']
			values = df['label'].value_counts(sort = False)
			table = table + values.to_string() + '\n\n'
			fig = generatePieChart(values,mlabels,restaurant,model)
			pie_Chart_Path = '.\\charts\\experiment_Charts\\pie_Charts\\'+restaurant + '_Pie_Chart_' + model
			#fig.savefig(pie_Chart_Path)
		print(table)
		
		table_Path = '.\\charts\\experiment_Charts\\data_Tables\\' + model + '_Experiment_Data_Table.txt'
		# with open(table_Path,'w') as file:
		# 	file.write(table)

## 2. Generate WordCloud
	# ## Transform documents to document term matrix using fit tfidf vectorizor  
	df = pd.DataFrame(list(db['processed_Training_Data'].find({},{'_id':0,'lema_text':1,'label':1})))
	vectorizor = joblib.load('.\\trained_Models\\TFIDF_Vectorizer.sav')
	v = vectorizor.transform(df['lema_text'].to_numpy())
	
	## Load and prepare mask image for word cloud generation
	mask2 = np.array(Image.open(".\\charts\\McD_logo.png").convert("RGB"))
	mask2[mask2 == 0] = 255

	## Use pandas Dataframe to calculate mean TFIDF for all words
	tframe = pd.DataFrame(v.toarray(),columns = vectorizor.get_feature_names())
	tframe.drop(labels = ['go','get'], axis = 1, inplace =True)
	tframe = tframe.T.mean(axis=1)

	#Include top 200 words from mean tfidf in cloud. Used McD logo colors and generate a higher res img than default.
	Cloud = WordCloud(background_color= (218,41,28), max_words=200,stopwords=STOPWORDS,mask = mask2,color_func=lambda *args, **kwargs: (255,199,44),collocations=False,height = 1600,width =1000).generate_from_frequencies(tframe) 
	plt.figure(figsize = (20, 10)) 
	plt.imshow(Cloud,interpolation="bilinear") 
	plt.axis("off")
	fig = plt.gcf()
	plt.show()
	#fig.savefig('.\\charts\\word_Cloud_McD.png')



## 3. Optimized MLP Performance (confusion matrix)
	X_train,X_test,y_train,y_test = train_test_split(v,df['label'],test_size=.2,random_state=2372017502)
	mlp = joblib.load('.\\trained_Models\\Optimized_Multi_Layer_Perceptron_Trained_Model.sav')
	pred = mlp.predict(X_test)
	CM = confusion_matrix(y_test,pred)
	cm_display = ConfusionMatrixDisplay(CM)
	cm_display.plot()
	fig = plt.gcf()
	plt.show()
	#fig.savefig('.\\charts\\Confusion_Matrix_MLP_Optimized.png')
	
	
# 	# y_score = mlp2.predict_proba(X_test)

# 	# prec, recall, _ = precision_recall_curve(y_test, y_score[:,1], pos_label=4)
# 	# pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
# 	# pr_display.plot()
# 	# plt.savefig('precison_recall.png')


# 	# fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=4)
# 	# roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
# 	# roc_display.plot()
# 	# plt.savefig('Roc_Curve.png')



# Get a Sample of tweets for review (human annotating) ##

	f1 = pd.DataFrame(sampleDB(db.exp_Data_Wendys_Combined,40))
	f2 = pd.DataFrame(sampleDB(db.exp_Data_BK_Combined,40))
	f3 = pd.DataFrame(sampleDB(db.exp_Data_McD_Combined,40))

	sample_Exper = pd.concat([f1,f2,f3])	
	# print(sample_Exper)
	sample_Exper = sample_Exper.sample(100)
	sample_Train = pd.DataFrame(sampleDB(db.processed_Training_Data,100))

	print(sample_Exper)
	print(sample_Train)

## Save them to a file ## 

# # Create a file that can be human annotated
# 	i = 1
# 	with open('.\\charts\\samples_To_Label\\sample_Train_Data_To_Label','w') as file:
# 		s = ''
# 		for ind in sample_Train.index:
# 			file.write(str(i)+ '. ')
# 			json.dump(sample_Train['text'][ind],file)
# 			file.write('     LABEL: ')
# 			json.dump(str(sample_Train['label'][ind]),file)
# 			file.write('\n\n')
# 			i = i+1

# 	i = 1
# 	with open('.\\charts\\samples_To_Label\\sample_Exper_Data_To_Label','w') as file:
# 		s = ''
# 		for ind in sample_Exper.index:
# 			file.write(str(i)+ '. ')
# 			json.dump(sample_Train['text'][ind],file)
# 			file.write('     LABEL: ')
# 			json.dump(str(sample_Train['label'][ind]),file)
# 			file.write('\n\n')
# 			i = i +1
