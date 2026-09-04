import pandas as pd
import sklearn.model_selection as ms
import sklearn.neighbors as n

df = pd.read_csv('data/heart/data_cleaned_up.csv')

y = df["num"]
x = df.drop(["num"], axis=1)

xtrain, xtest, ytrain, ytest = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42)

for k in range(2,15):
    model = n.KNeighborsClassifier(n_neighbors=k)
    model.fit(xtrain, ytrain)

    train_score = model.score(xtrain, ytrain)
    test_score = model.score(xtest, ytest)

    print(train_score)
    print(test_score)

ypred = model.predict(xtest)
print(ypred)


