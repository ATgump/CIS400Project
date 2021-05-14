from ourCorpus import batch_lemmatizer
import pymongo
import pandas as pd
import matplotlib.pyplot as plt
import random
import json


if __name__ == "__main__":
	# string = '@Sparkywoomy A freshsalad of all fruits &amp; vegetables is a part of the McSparky meald availableee noaw at your nearest McDonalds staore. #trending 😠😠😠 https://t.co/KhWTXElnJb &gt; (a $100 value) :)'
	# batch_lemmatizer([string])

	
	client = pymongo.MongoClient('mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true')
	db = client['tweet_DB']

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





	## Sample tweets for review ##
	f1 = pd.DataFrame(db.exp_Data_Wendys_Combined.aggregate([{'$sample':{'size':40}}]))
	f2 = pd.DataFrame(db.exp_Data_BK_Combined.aggregate([{'$sample':{'size':40}}]))
	f3 = pd.DataFrame(db.exp_Data_McD_Combined.aggregate([{'$sample':{'size':40}}]))

	sample_Exper = pd.concat([f1,f2,f3])#pd.concat([pd.DataFrame(db.exp_Data_Wendys_Combined.aggregate([{'$sample':{'size':40}}]),pd.DataFrame(db.exp_Data_BK_Combined.aggregate([{'$sample':{'size':40}}]))),(pd.DataFrame(db.exp_Data_McD_Combined.aggregate([{'$sample':{'size':40}}])))])
	print(sample_Exper)
	sample_Exper = sample_Exper.sample(100)
	sample_Train = pd.DataFrame(db.processed_Training_Data_Three.aggregate([{'$sample':{'size':100}}]))




	i = 1
	with open('sample_Train_Data_To_Label','w') as file:
		s = ''
		for ind in sample_Train.index:
			file.write(str(i)+ '. ')
			json.dump(sample_Train['text'][ind],file)
			file.write('     LABEL: ')
			json.dump(str(sample_Train['label'][ind]),file)
			file.write('\n\n')
			i = i+1

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


