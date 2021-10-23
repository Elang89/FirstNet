import tensorflow as tf

from app.models.nn import FirstNet
from app.services.trainer import Trainer

def main() -> None: 
    mnist = tf.keras.datasets.mnist

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    model = FirstNet()
    trainer = Trainer(model)
    trainer.compile_model()
    trainer.train_model(X_train, y_train)
    trainer.evaluate(X_test, y_test)



if __name__ == "__main__":
    main()