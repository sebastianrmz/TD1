# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# load the dataset
dataset = loadtxt('TD1_keras_data.txt', delimiter=',')
# split into input (V) and output (s) variables
V = dataset[:,0:2]
s = dataset[:,2]


# define the keras model
model = Sequential()
model.add(Dense(1, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='tanh'))

# compile the keras model
model.compile(optimizer='Adam', loss='mean_absolute_error',metrics=["mse"])
#model.summary()

# train the neural network
model.fit(V, s, epochs=1500, batch_size=89, verbose=0)

# query the neural network with original signal
prediction = model.predict(V,verbose=0)

# display the results
for i in range(89):
	print('%s => %f (expected %f) (error %f)' % (V[i].tolist(), prediction[i], s[i], s[i] - prediction[i]))