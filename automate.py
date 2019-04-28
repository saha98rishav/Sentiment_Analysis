import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Taking CSV from user
csv_file_name = input("Enter the name of the dataset : ")
csv_file_name = csv_file_name + ".csv"

data = pd.read_csv(csv_file_name)
print(data.head(1))
# score = input("Enter the name of attribute containing the 'Scores' : ")
text = input("Enter the name of the attribute containing the 'Reviews' : ")
rows = int(input("Enter the number of rows for processing : "))
data = pd.read_csv(csv_file_name, nrows = rows, usecols = [text])
df = pd.DataFrame(data)

# cleaning the texts
corpus = []

for i in tqdm(range(0, rows)):
    review = re.sub('[^a-zA-Z]', ' ',df[text][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

    # Create Bag of Words model
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()


from sklearn.externals import joblib
model = joblib.load('model_joblib')
predict = model.predict(X)


(good, bad, neutral) = (0,0,0)
for x in predict:
    if x=='good':
        good = good + 1
    elif x=='bad':
        bad = bad + 1
    elif x=='neutral':
        neutral = neutral + 1
    print(x)


print("Good = ",good)
print("Neutral = ",neutral)
print("bad = ",bad)

import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Good', 'Neutral', 'Bad'
sizes = [good, neutral, bad]
explode = (0, 0, 0)  # only "explode" the 1st slice (i.e. 'Good')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
