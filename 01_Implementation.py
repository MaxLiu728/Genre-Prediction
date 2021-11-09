# Have added Git
# Set Working environment
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

'''
The next few codes will serve for the experiment purpose
'''

df_1= pd.read_csv('data.csv')
b= df_1['Lyric'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

stop_words = set(stopwords.words('english'))
b= df_1['Lyric'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

## Tokenize
b_2= b.apply(lambda x: word_tokenize(x))


# Use pos_tag to get the type of the world and then map the tag to the format wordnet lemmatizer would accept.
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = WordNetLemmatizer()

b_3= b_2.apply(lambda x: ' '.join([lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in x]))




stop_words_2= ('10','100','20','2x','3x','4x','50','im')
stop_words.add(x for x in stop_words_2)
stop_words.add('10')
for i in ('10','100','20','2x','3x','4x','50'):
    stop_words.add(i)



cv_2= CountVectorizer(stop_words=stop_words, min_df=300, lowercase= True, ngram_range=(1,1))

transform_temp= cv_2.fit_transform(b_3)


df_output= pd.DataFrame(transform_temp.toarray(),columns= cv_2.get_feature_names_out())


# Split Train Validation Test
y_temp= df_1.Genre
y_temp_test= y_temp[-5000: ]
y_temp_train= pd.DataFrame(y_temp[:50000], columns = ['Genre'])

x_temp_train= pd.DataFrame(df_output.iloc[:50000,:], columns = cv_2.get_feature_names_out())
x_temp_test = pd.DataFrame(df_output.iloc[-5000:,:], columns = cv_2.get_feature_names_out())


df_overall = pd.concat([x_temp_train, y_temp_train], axis =1 )

# Select nlargest frequencies word for each class
df_overall.columns[:-1]

## Some plots
sns.catplot(x= 'Genre',kind= 'count', data =y_temp_train )

temp_5= pd.DataFrame(np.ones((3,1)),columns=['exm'],index = ['Hip Hop','Pop','Rock'])

for index,value in enumerate (df_overall.columns[:-1]):
    temp_5[value]= pd.DataFrame(df_overall.groupby('Genre')[value].sum())

p= temp_5.apply(lambda s,n: s.nlargest(n).index, axis =1 ,n=15)



    temp_3= pd.concat([pd.DataFrame(df_overall.groupby('Genre')[i].sum())], axis=1)


    print(df_overall.groupby('Genre')[i].mean())

temp_2= pd.DataFrame(df_overall.groupby('Genre')['knife'].mean())

df_overall.groupby('Genre').iloc[:,2].mean()
df_overall.apply(lambda x: df_overall.groupby('Genre').x.mean(), axis=1)

# Observe
df_1.Genre


# Model
model = RandomForestClassifier(criterion='entropy',n_estimators=100)
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=66)
n_scores = cross_val_score(model, x_temp_train, y_temp_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')







'''
The next few codes will walk through the whole process. 
'''
# Read data
def load_data():
    data = pd.read_csv('data.csv')
    X = data['Lyric']
    y = data['Genre']
    return X,y

def transorm_data():
    X,y= load_data()
    # Data Preprocessing
    ## Text cleaning- Remove punctuations
    X = X.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    ## Tokenize and Lemmatization
    X_tokenize = X.apply(lambda x: word_tokenize(x))
    ## Join
    lemmatizer = WordNetLemmatizer()
    X_lemmatize = X_tokenize.apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in x]))
    ## Countvectorize
    for i in ('10', '100', '20', '2x', '3x', '4x', '50', ):
        stop_words.add(i) # Add more stop words

    CV = CountVectorizer(stop_words=stop_words, min_df=300, lowercase=True, ngram_range=(1, 1))
    X_vectorized =pd.DataFrame(X_lemmatize.toarray(),columns= cv.get_feature_names_out())
    return X_vectorized,y

def visualize (X,y):
    '''

    :param X: X is the countvectorized words for all Lyrics
    :param y: y is the Genre classifications
    :return: The barchart for y, as well as the words for top frequencies for each type of Genre
    '''
    print(sns.catplot(x= 'Genre',kind= 'count', data =pd.DataFrame(y[:50000], column= ['Genre'])))


def k_fold(x,y,k):
    '''

    :param x: x is the features
    :param y: response variable
    :param k: k is the number of folds
    :return: average f1score metric
    '''

def main():
    # Load data
    X_input, y_input = transorm_data()

    ## Split train and test data set
    y_train = pd.DataFrame(y_input[:50000], columns=['Genre'])
    y_test = pd.DataFrame(y_input[-5000:], columns=['Genre'])

    x_train = pd.DataFrame(X_input[:50000], columns=CV.get_feature_names_out())
    x_test = pd.DataFrame(X_input[-5000:], columns=CV.get_feature_names_out())


if __name__ == "__main__":
    main()
