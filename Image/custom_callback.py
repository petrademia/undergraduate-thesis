from keras.callbacks import Callback
from time import time

class TimingCallback(Callback):
    def on_train_begin(self, logs={}):
        self.logs = []
        
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()
        
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time() - self.starttime)
