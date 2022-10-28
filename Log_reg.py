from unittest import result
import pandas as pd
import numpy as np

train_df = pd.read_csv("/home/sirabas/MNIST/train.csv")
test_df = pd.read_csv("/home/sirabas/MNIST/test.csv")
X_train = train_df.drop('label',axis=1)
y_train = train_df['label']
X_test= test_df

X_train = X_train/255
X_test = X_test/255
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(fit_intercept=True, multi_class='auto', penalty='l2', solver='sag',max_iter=1000,C=50)

logreg.fit(X_train, y_train)

y_predict=logreg.predict(X_test)

image_id = pd.Series(range(1,28001),name='ImageId')
y_preds = pd.Series(y_predict,name = 'Label')
pred = pd.concat([image_id,y_preds],axis=1)
pred.to_csv('submission.csv',index=False)
