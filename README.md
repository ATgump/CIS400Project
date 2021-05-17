# CIS400Project - Sentiment Analysis of Restuarants
#### By: Avery Gump, Jacob Morrison, Chi Chi Tong, and Jon Williams
Code found at: https://github.com/ATgump/CIS400Project  
If you run into any troubles contact me at: atgump@syr.edu
Python version used: 3.9.5  
To install the required packages for our project use: 
```powershell 
pip install -r './requirements.txt'
```
You may also need to download nltk data (stopwords). Follow this guide: https://www.nltk.org/data.html





To connect to our MongoDB use the connection string: mongodb+srv://CISProjectUser:U1WsTu2X6fix49PA@cluster0.ttjkp.mongodb.net/test?authSource=admin&replicaSet=atlas-vvszkk-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true

Most files can be run as is and will show some demo of what each file is doing. See notes for information about running each file.
## Python Files
### trainDataCollector.py
While most exceptions are handled some may still occur, it was infrequent enough, for our needs, to not handle and just restart the program. I commented out the part that adds the tweets to mongo so you can test without altering the data set we collected.

Collection query: `["McDonalds","Wendys",'Burger King','Pizza Hut',"Whataburger","In-N-OutBurger","White Castle","Starbucks","Auntie Anne\'s","Popeyes","Chick-fil-A",'Taco Bell',"Arby\'s","Dairy Queen"]`

Status fields saved to mongo: text (or full_text), entities, lang, id_str

Filter out retweets with:
```python
if hasattr(status,'retweeted_status')
    return
```
### trainDataLabeler.py
I ran into out of memory issues when trying to use too many threads even on a machine with 32 gb of ram, so, I set the number of jobs to 1 to make sure this doesn't happen on the first test run. With that being said, you may want to increase this number to speed up the labeling. To do this change n_jobs in the following line(34) of code: 
```python 
executor = Parallel(n_jobs=1, backend='multiprocessing', prefer="processes")
```
Also, I commented out the part that adds the tweets to mongo so you can test without altering the data set we labeled. 
### tweetPreprocessor.py
Memory issues same as trainDataLabeler (except n_jobs in line 148). Also, if you would like to see each step in the preprocessing, uncomment all the print statements in chunk_processor. I have provided a sample tweet that shows each step in the preprocessing (commented at bottom).
### trainModels.py
We commented out the the pieces of code where the models/vectorizor/report are saved because these files are already included in the trained_Models directory. If you would like to make new ones delete the trained_Models directory and uncomment the following lines of code:
```python 
#joblib.dump(vectorizor, TF_Vec)
```
```python 
# joblib.dump(trained, path_Model)
```
```python 
# with open(path_Report,'w') as file:
#  	file.write(report)
```
The models were only trained on the state that is currently in the file: 2372017502. We understand this is sub-optimal and it would have been preferred to have trained multiple times using a random state and taken the average, but, with the limited time we have and the time it takes to train some of the models on such a large dataset we decided against it. 
### gridSearchOptimizer.py
We left in some of the parameters that were tested commented out, some are uncommented so that you can test the program. We did not run into out of memory issues on this part so I left n_jobs = -1 (all cores used) but if you do run into this problem just change that to somthing like 2-3. This is the line of code to change(56): 
```python
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3,verbose = 100)	
``` 
### experimentDataCollector.py
Similar to trainData collector but with search query: `["McDonalds","Wendys","Burger King"]` and instead of adding all the tweets to a single database it seperates them depending on which restaurant they are about.
### experimentDataLabeler.py
Adding to DB is commented out if you would like to add the data to DB uncomment the following line:
```python
# db[ins].insert_many(labeled)
```

 n_jobs = -1 as we didn't run into memory issues at this stage, but, if you do, change n_jobs to 2 or 3 (line 77) .  
### chartGeneration.py
Charts/figures generated here are saved in the charts directory. Saving the files to this directory has been commented out throughout. It will still generate the graphs/charts but not save them to the charts directory.
1. Create Pie Charts from DB random sample
2. Create Word cloud from mean TFIDF
3. Create Confusion Matrix for Optimized Multi-Layer Perceptron 
4. Get sample from experiment/training data for human annotation (ran out of time to annotate)

## Mongo Collections
1. unprocessed_Tweets: Tweets collected for training our Models (101,879)
2. labeled_Training_Data: labeled unprocessed_Tweets using TextBlob/VADER
3. processed_Training_Data: labeled tweets preprocessed
4. experimental_Data_RESTAURANT: tweets collected to conduct our experiment
5. exp_Data_RESTAURANT_MODEL: holds tweets that were labeled using our trained models for our experiment
