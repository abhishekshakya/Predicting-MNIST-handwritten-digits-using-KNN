import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("seaborn")


df = pd.read_csv('train.csv')
# print(df.shape)
# print(df.columns)

Y = df['label'].values
X = df.iloc[:,1:].values

# print(X.shape)
# print(Y.shape)

split = int(0.8*X.shape[0])

X_train = X[:split,:]
Y_train = Y[:split]

X_test = X[split:,:]
Y_test = Y[split:]


#visualising
def drawImg(sample):
	sample = sample.reshape((28,28))
	plt.imshow(sample)
	plt.show()

# drawImg(X_train[1])#image of zero
# drawImg(X_train[24])#image of two



##-------------------------------------------------KNN ALGO------------------------
def distance(x,point):
	d = np.sqrt(sum((x-point)**2))
	return d

def knn(X,Y,point,k=5):
	dist = []
	m = X.shape[0]

	for i in range(m):
		d = distance(X[i],point)
		dist.append((d,Y[i]))

	dist = sorted(dist,key= lambda d: d[0])
	dist = np.array(dist[:k])

	#adding voting part
	classes = np.unique(np.array(Y))
	# print(classes)

	votes = np.zeros(len(classes))

	for d in dist:#farther points will contribute less in voting part
		votes[int(d[1])]+= 1/(d[0])

	# print(votes)
	pred = np.argmax(votes)

	return pred



#-----------------------------------------------------------------------------------------------


y_pred = [] 

for test in X_test[:10,:]:
	y_pred.append(knn(X_train,Y_train,test,k=5))

count = 0;
for i in range(len(y_pred)):
	if y_pred[i]==Y_test[i]:
		count+=1
accuracy = count/len(y_pred)

print(f"accuracy: {accuracy*100} (this if for 10 examples)")

##it takes a lot of consuming power



#random example
y_pred = knn(X_train,Y_train,X_test[58],k=5)
print(y_pred)
drawImg(X_test[58])