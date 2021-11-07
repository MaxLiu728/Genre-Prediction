# Have added Git

import pandas as pd
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import string

df_1= pd.read_csv('data.csv')
b= df_1['Lyric'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

cv= CountVectorizer(stop_words=['a','you','to','me','12'], min_df=2)
transform_temp= cv.fit_transform(b)

transform_temp.toarray().shape
cv.get_feature_names()

df_output= pd.DataFrame(transform_temp.toarray(),columns= cv.get_feature_names())

df_output