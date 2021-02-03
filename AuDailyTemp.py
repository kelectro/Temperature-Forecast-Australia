import numpy as np
import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image  as mpimg

tf.keras.backend.clear_session()  #clear any internal variable
tf.random.set_seed(51)
np.random.seed(51)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def plot_series(time, series, color, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format, color=color)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def window_data(series, batch_size, window_size, shuffle_buffer):
    data = tf.expand_dims(series, axis =-1)
    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.window(size = window_size+1 , shift=1, drop_remainder=True)
    data = data.flat_map(lambda w: w.batch(window_size+1))
    data = data.shuffle(shuffle_buffer)
    data = data.map(lambda w: (w[:-1], w[1:]))
    return data.batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

# Parse data
path = '/home/kiagkons/tensorflow_preparation/Australia Weather/daily_min_temp.csv'
temp = []
index = []
i= 0
with open(path,'r') as file:
    data = csv.reader(file)
    next(data)
    for row in data:
        temp.append(float(row[1]))
        index.append(i)
        i+=1

index = np.array(index)
temp = np.array(temp)
'''
# Inspect for seasonality, trend, noise
plt.figure(figsize=(10,6))
plt.plot(index,temp)
plt.show()
'''

split = 2500
x_train = temp[:split]
time_train = index[:split]
x_valid = temp[split:]
time_valid = index[split:]

batch_size = 100
shuffle_buffer = 1000
window_size = 60

train_data = window_data(x_train,
                         window_size=window_size,
                         batch_size=batch_size,
                         shuffle_buffer=shuffle_buffer)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv1D(filters=60,kernel_size=5,
                           padding='causal',
                           activation='relu',
                           input_shape=[None,1]),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60,return_sequences=True),
    # tf.keras.layers.LSTM(60,return_sequences=True),
    tf.keras.layers.Dense(30, activation='relu'),
    # tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),

    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*400)
    ])

sgd = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=sgd,
              metrics=['mae'])

history = model.fit(train_data,
          epochs=400)

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 0.01, 0, 60])
# plt.show()


rnn_forecast = model_forecast(model, temp[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split - window_size:-1, -1, 0]

print('Mean average error is :',tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid, color='b')
plot_series(time_valid, rnn_forecast, color='r')
plt.show()


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
loss=history.history['loss']

epochs=range(len(loss)) # Get number of epochs


#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r')
plt.title('Training loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])

plt.figure()

plt.show()

zoomed_loss = loss[250:]
zoomed_epochs = range(250,400)


#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(zoomed_epochs, zoomed_loss, 'r')
plt.title('Training loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss"])

plt.figure()
plt.show()
