#DSCI-633 FALL-22 PROJECT SUBMISSION BY JOHN MELWIN RICHARD
import pandas as pd
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.utils import resample
import warnings
import sys

##################################
sys.path.insert(0, '../..')
##################################
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.options.display.max_colwidth = 100
warnings.filterwarnings("ignore")

class my_model():
    def sampling(self, X_train, y_train):
        # up-sampling the minority class
        sampled_data = X_train
        sampled_data = sampled_data.assign(fraudulent = y_train)
        spam = sampled_data[sampled_data["fraudulent"] == 0]
        real = sampled_data[sampled_data["fraudulent"] == 1]
        spam_upsample = resample(real,replace=True,n_samples= round(0.5*len(real)),random_state=42)
        data_upsampled = pd.concat([spam, spam_upsample])
        data_upsampled = shuffle(data_upsampled)
        y = data_upsampled["fraudulent"]
        data_upsampled = data_upsampled.drop(['fraudulent'], axis=1)
        split_point = int(0.8 * data_upsampled.shape[0])
        X_train = data_upsampled.iloc[:split_point]
        y_train = y.iloc[:split_point]
        return X_train,y_train

    def fit(self, X_train, y_train):
        # pre-processing X_train data
        #X_train, y_train = self.sampling(X_train, y_train)
        X_train = self.pre_process(X_train)
        # vectorizing X_train data
        self.vectorizer = TfidfVectorizer( max_features= 6500, decode_error='replace', encoding='utf-8')
        self.vectorizer.fit(X_train['text'].values.astype('U'))
        pre_processed_X = self.vectorizer.fit_transform(X_train['text'].values.astype('U')).toarray()  # independent
        X_train = pd.DataFrame(pre_processed_X, columns=self.vectorizer.get_feature_names_out())
        # y_train data
        y_train = pd.DataFrame(y_train)
        y_train.columns = ['fraudulent']
        #---------------------------------------------------------------------------------------------------
        # PAC
        pac = PassiveAggressiveClassifier(random_state= False, loss = "squared_hinge") # new model (performing very good for text data compared to other models)
        # loss = ['hinge', 'squared_hinge']//best loss = squared hinge
        # shuffle = [True, False] // best shuffle = false
        # param_dist = { "n_jobs": [-1]} // use all cores
        # grid_search = GridSearchCV(pac, param_grid=param_dist, scoring='f1', cv=5)
        pac.fit(X_train, y_train)
        self.best_pac = pac

    def predict(self, X):
        # pre-processing X_test data
        X_test = self.pre_process(X)
        # apply same vectorizing to X_test data
        X_test = self.vectorizer.transform(X_test['text'].values.astype('U')).toarray()
        X_test = pd.DataFrame(X_test, columns=self.vectorizer.get_feature_names_out())
        y_predict_pac = self.best_pac.predict(X_test)
        return  y_predict_pac

    def pred_probs(self, X_test):
        X_test = self.pre_process(X_test)
        # apply same vectorizing to X_test data
        X_test = self.vectorizer.transform(X_test['text'].values.astype('U')).toarray()
        X_test = pd.DataFrame(X_test, columns=self.vectorizer.get_feature_names_out())
        predict_probs = self.best_pac.predict_proba(X_test)
        return predict_probs

    def pre_process(self, X):
        data = X
        clean_data = data
        clean_data['location'] = clean_data['location'].fillna("no")  # can use " " instead of unknown for improvement
        clean_data['requirements'] = clean_data['requirements'].fillna("no")  # can use " " instead of unknown for nothing
        # combining every text column to one column named 'text'
        clean_data['text'] = clean_data['title'] + ' ' + clean_data['location'] + ' ' + clean_data['description'] + ' ' + clean_data['requirements']  # can add '-' inbetween for improvement check
        clean_data = clean_data.drop(['title', 'location', 'description', 'requirements'], axis=1)
        clean_data.fillna('', inplace=True)
        # Remove Stopwords, Punctuation and Special Characters
        clean_data['text'] = clean_data['text'].str.lower()
        clean_data['text'] = clean_data['text'].apply(lambda x: gensim.parsing.preprocessing.remove_stopwords("".join(x)))
        clean_data['text'] = clean_data['text'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')
        clean_data['text'] = clean_data['text'].str.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '')
        clean_data['text'] = clean_data['text'].str.replace(r'[\'-]', '')
        clean_data['text'] = clean_data['text'].str.replace(r'[\'\",()*&^%$#@!~`+=|/<>?{}\[\]\/\\:;\_]]',' ')  # remove punctuation
        clean_data['text'] = clean_data['text'].apply(lambda x: ' '.join([word for word in x.split() if not word.startswith('url')]))
        clean_data['text'] = clean_data['text'].apply(lambda x: ' '.join([word for word in x.split(' ') if len(word) < 25]))
        #clean_data['text'] = clean_data['text'].map(lambda x: re.sub(r'\W+', ' ', x))
        clean_data['text'] = clean_data['text'].str.replace(r'[0-9]', '')  # get rid of numbers
        clean_data['text'] = clean_data['text'].str.replace(r'[^a-z]', ' ')  # get rid of any non english characters
        clean_data.drop(['telecommuting', 'has_company_logo', 'has_questions'], axis=1, inplace=True)
        return clean_data