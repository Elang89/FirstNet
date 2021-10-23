import tensorflow as tf

from tensorflow.keras import Model

class Trainer:

    def __init__(self, model: Model):
        self.model = model 

    def compile_model(self):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer="RMSProp", loss=loss_fn, metrics=["accuracy"])  

    def train_model(self, X_train, y_train): 
        self.model.fit(X_train, y_train, epochs=5)

    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test, verbose=2)
