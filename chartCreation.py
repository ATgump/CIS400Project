from tweetPreprocessor import batch_lemmatizer
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
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
	print(values)
	s = restaurant + '_Pie_Chart_' + model
	print(s)
	plt.show()
	return fig

if __name__ == "__main__":
	
	## Connection to Mongo tweet_DB
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']


	## Create Pie charts for different Experiments (models/restaurants)
	restaurants = ['Wendys','BK','McD']
	models = ['MLP','RF','LSV','Combined']

	for model in models:
		for restaurant in restaurants:
			col = 'exp_Data_' + restaurant + '_' + model
			if model == 'Combined':
				sample = sampleDB(col,2000)
			else:
				sample = sampleDB(col,2500)	
			df = pd.DataFrame(sample)
			mlabels = ['negative','neutral','positive']
			values = df['label'].value_counts(sort = False)
			fig = generatePieChart(values,mlabels,restaurant,model)
			s = restaurant + '_Pie_Chart_' + model
			fig.savefig(s)


## Generate WordCloud
	## Transform documents to document term matrix using fit tfidf vectorizor  
	df = pd.DataFrame(list(db['processed_Training_Data_Three'].find({},{'_id':0,'lema_text':1,'label':1})))
	vectorizor = joblib.load('TFIDF_Vectorizer.sav')
	v = vectorizor.transform(df['lema_text'].to_numpy())
	
	## Load and prepare mask image for word cloud generation
	mask2 = np.array(Image.open("McD_logo.png").convert("RGB"))
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
	plt.show()




## Optimized MLP Performance (confusion matrix)
	X_train,X_test,y_train,y_test = train_test_split(v,df['label'],test_size=.2,random_state=2372017502)
	mlp = joblib.load('Optimized_Multi_Layer_Perceptron_Trained_Model.sav')
	pred = mlp.predict(X_test)
	CM = confusion_matrix(y_test,pred)
	cm_display = ConfusionMatrixDisplay(CM)
	cm_display.plot()
	plt.savefig('Confusion_Matrix_MLP_Optimized.png')



## Sample tweets for review ##
	f1 = pd.DataFrame(sampleDB(db.exp_Data_Wendys_Combined,40))
	f2 = pd.DataFrame(sampleDB(db.exp_Data_BK_Combined,40))
	f3 = pd.DataFrame(sampleDB(db.exp_Data_McD_Combined,40))

	sample_Exper = pd.concat([f1,f2,f3])	
	# print(sample_Exper)
	sample_Exper = sample_Exper.sample(100)
	sample_Train = pd.DataFrame(db.processed_Training_Data_Three,100)

## Create a file that can be human annotated
	# i = 1
	# with open('sample_Train_Data_To_Label','w') as file:
	# 	s = ''
	# 	for ind in sample_Train.index:
	# 		file.write(str(i)+ '. ')
	# 		json.dump(sample_Train['text'][ind],file)
	# 		file.write('     LABEL: ')
	# 		json.dump(str(sample_Train['label'][ind]),file)
	# 		file.write('\n\n')
	# 		i = i+1

	# i = 1
	# with open('sample_Exper_Data_To_Label','w') as file:
	# 	s = ''
	# 	for ind in sample_Exper.index:
	# 		file.write(str(i)+ '. ')
	# 		json.dump(sample_Train['text'][ind],file)
	# 		file.write('     LABEL: ')
	# 		json.dump(str(sample_Train['label'][ind]),file)
	# 		file.write('\n\n')
	# 		i = i +1





### SAMPLE MLP/RF/LSV/Combined

	# sample_Wendys_MLP = db.exp_Data_Wendys_MLP.aggregate([{'$sample':{'size':2500}}])
	# sample_BK_MLP = db.exp_Data_BK_MLP.aggregate([{'$sample':{'size':2500}}])
	# sample_McD_MLP = db.exp_Data_McD_MLP.aggregate([{'$sample':{'size':2500}}])

	# sample_Wendys_RF = db.exp_Data_Wendys_RF.aggregate([{'$sample':{'size':2500}}])
	# sample_BK_RF = db.exp_Data_BK_RF.aggregate([{'$sample':{'size':2500}}])
	# sample_McD_RF = db.exp_Data_McD_RF.aggregate([{'$sample':{'size':2500}}])

	# sample_Wendys_LSV = db.exp_Data_Wendys_LSV.aggregate([{'$sample':{'size':2500}}])
	# sample_BK_LSV = db.exp_Data_BK_LSV.aggregate([{'$sample':{'size':2500}}])
	# sample_McD_LSV = db.exp_Data_McD_LSV.aggregate([{'$sample':{'size':2500}}])

	# sample_Wendys_Combined = db.exp_Data_Wendys_Combined.aggregate([{'$sample':{'size':2000}}])
	# sample_BK_Combined = db.exp_Data_BK_Combined.aggregate([{'$sample':{'size':2000}}])
	# sample_McD_Combined = db.exp_Data_McD_Combined.aggregate([{'$sample':{'size':2000}}])

## Create Data frames for plotting

	# df_Wendys_MLP = pd.DataFrame(sample_Wendys_MLP)
	# df_BK_MLP = pd.DataFrame(sample_BK_MLP)
	# df_McD_MLP = pd.DataFrame(sample_McD_MLP)

	# df_Wendys_RF = pd.DataFrame(sample_Wendys_RF)
	# df_BK_RF = pd.DataFrame(sample_BK_RF)
	# df_McD_RF = pd.DataFrame(sample_McD_RF)

	# df_Wendys_LSV = pd.DataFrame(sample_Wendys_LSV)
	# df_BK_LSV = pd.DataFrame(sample_BK_LSV)
	# df_McD_LSV = pd.DataFrame(sample_McD_LSV)

	# df_Wendys_Combined = pd.DataFrame(sample_Wendys_Combined)
	# df_BK_Combined = pd.DataFrame(sample_BK_Combined)
	# df_McD_Combined = pd.DataFrame(sample_McD_Combined)

	# print(df_wendys)
	#print(df_BK_MLP['label'].value_counts(sort = False))


	# mlabels = ['negative','neutral','positive']

### MLP PIE CHARTS ###
	# plt.pie(df_Wendys_MLP['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_Wendys_MLP['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('Wendys_Pie_Chart_MLP')


	# plt.pie(df_BK_MLP['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_BK_MLP['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('BK_Pie_Chart_MLP')


	# plt.pie(df_McD_MLP['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_McD_MLP['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('McD_Pie_Chart_MLP')


### RF Pie charts ######

	# plt.pie(df_Wendys_RF['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_Wendys_RF['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('Wendys_Pie_Chart_RF')


	# plt.pie(df_BK_RF['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_BK_RF['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('BK_Pie_Chart_RF')


	# plt.pie(df_McD_RF['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_McD_RF['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('McD_Pie_Chart_RF')


#### LSV Pie Charts #####

	# plt.pie(df_Wendys_LSV['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_Wendys_LSV['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('Wendys_Pie_Chart_LSV')


	# plt.pie(df_BK_LSV['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_BK_LSV['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('BK_Pie_Chart_LSV')


	# plt.pie(df_McD_LSV['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_McD_LSV['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('McD_Pie_Chart_LSV')


### Combined Pie Charts ###

	# plt.pie(df_Wendys_Combined['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_Wendys_Combined['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('Wendys_Pie_Chart_Combined')


	# plt.pie(df_BK_Combined['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_BK_Combined['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('BK_Pie_Chart_Combined')


	# plt.pie(df_McD_Combined['label'].value_counts(sort = False),labels=mlabels)
	# fig = plt.gcf()
	# print(df_McD_Combined['label'].value_counts(sort = False))
	# plt.show()
	# fig.savefig('McD_Pie_Chart_Combined')








