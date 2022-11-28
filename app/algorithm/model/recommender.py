import numpy as np, pandas as pd
import os
from sklearn.utils import shuffle
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


MODEL_NAME = "recommender_base_matrix_factorizer_using_sgd"

model_params_fname = "model_params.save"
model_wts_fname = "model_wts.save"
history_fname = "history.json"


COST_THRESHOLD = float("inf")

print("GPU devices = ", tf.config.list_physical_devices("GPU"))


class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get("loss")
        if loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val):
            print("\nCost is inf, so stopping training!!")
            self.model.stop_training = True


class Recommender:
    def __init__(
        self, N, M, K=10, l2_reg=1e-5, lr=0.08, momentum=0.99, batch_size=512, **kwargs
    ):
        self.N = N
        self.M = M
        self.K = K
        self.l2_reg = l2_reg
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

        self.model = self.build_model()
        self.model.compile(
            loss="mse",
            # optimizer=Adam(learning_rate=self.lr),
            optimizer=SGD(learning_rate=self.lr, momentum=self.momentum),
            metrics=["mae"],
        )

    def build_model(self):
        u = Input(shape=(1,))
        m = Input(shape=(1,))
        u_embedding = Embedding(self.N, self.K, embeddings_regularizer=l2(self.l2_reg))(
            u
        )  # shape => (N, 1, K)
        m_embedding = Embedding(self.M, self.K, embeddings_regularizer=l2(self.l2_reg))(
            m
        )  # shape => (N, 1, K)

        u_bias = Embedding(self.N, 1, embeddings_regularizer=l2(self.l2_reg))(
            u
        )  # shape => (N, 1, 1)
        m_bias = Embedding(self.M, 1, embeddings_regularizer=l2(self.l2_reg))(
            m
        )  # shape => (N, 1, 1)
        x = Dot(axes=2)([u_embedding, m_embedding])  # shape => (N, 1, 1)

        x = Add()([x, u_bias, m_bias])
        x = Flatten()(x)  # shape => (N, 1)

        model = Model(inputs=[u, m], outputs=x)
        return model

    def fit(self, X, y, validation_split=None, epochs=25, verbose=0):

        early_stop_loss = "val_loss" if validation_split is not None else "loss"
        early_stop_callback = EarlyStopping(
            monitor=early_stop_loss, min_delta=1e-4, patience=3
        )
        infcost_stop_callback = InfCostStopCallback()

        history = self.model.fit(
            x=[X[:, 0], X[:, 1]],
            y=y,
            validation_split=validation_split,
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=True,
            callbacks=[early_stop_callback, infcost_stop_callback],
        )
        return history

    def predict(self, X):
        preds = self.model.predict([X[:, 0], X[:, 1]], verbose=0)
        return preds

    def summary(self):
        self.model.summary()

    def evaluate(self, x_test, y_test):
        """Evaluate the model and return the loss and metrics"""
        return self.model.evaluate(x=[x_test[:, 0], x_test[:, 1]], y=y_test, verbose=0)

    def save(self, model_path):
        model_params = {
            "N": self.N,
            "M": self.M,
            "K": self.K,
            "l2_reg": self.l2_reg,
            "lr": self.lr,
            "momentum": self.momentum,
            "batch_size": self.batch_size,
        }
        joblib.dump(model_params, os.path.join(model_path, model_params_fname))

        self.model.save_weights(os.path.join(model_path, model_wts_fname))

    @staticmethod
    def load(model_path):
        model_params = joblib.load(os.path.join(model_path, model_params_fname))
        mf = Recommender(**model_params)
        mf.model.load_weights(
            os.path.join(model_path, model_wts_fname)
        ).expect_partial()
        return mf


def get_data_based_model_params(X):
    """
    returns a dictionary with N: number of users and M = number of items
    This assumes that the given numpy array (X) has users by id in first column,
    and items by id in 2nd column. the ids must be 0 to N-1 and 0 to M-1 for users and items.
    """
    N = int(X[:, 0].max() + 1)
    M = int(X[:, 1].max() + 1)
    return {"N": N, "M": M}


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    try:
        model = Recommender.load(model_path)
    except:
        raise Exception(
            f"""Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?"""
        )
    return model


def save_training_history(history, f_path):
    hist_df = pd.DataFrame(history.history)
    hist_json_file = os.path.join(f_path, history_fname)
    with open(hist_json_file, mode="w") as f:
        hist_df.to_json(f, indent=2)
