import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf

class LossHistory(tf.keras.callbacks.Callback): 
    
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('mse')
        plt.legend(loc="upper right")
        plt.savefig("mnist_keras.png")
        plt.show()
        
history = LossHistory()
model = Sequential()
model.add(ConvLSTM2D(64,kernel_size=(3,3),strides=(1, 1), padding='same', input_shape=(None, trainX.shape[2],trainX.shape[3], 2), return_sequences=True))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))#BN
model.add(ConvLSTM2D(32, kernel_size=(3,3), strides=(1, 1), padding='same',return_sequences=True))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))#BN
model.add(ConvLSTM2D(filters = 1, kernel_size=(3,3), strides=(1, 1), padding='same',return_sequences=False))
model.compile(loss="mae", optimizer='adam',metrics=['mse'])

model.fit(trainX, trainY, epochs=300, batch_size=4, verbose=2, callbacks=[history,checkpointer],validation_data=(validX,validY))
history.loss_plot('epoch')
