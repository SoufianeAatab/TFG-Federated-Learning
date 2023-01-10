from keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import socket
import time
import math
(X_train, y_train), (X_test, y_test) = mnist.load_data()

onehot_encoder = OneHotEncoder(sparse=False)

y_train_one_hot=onehot_encoder.fit_transform(y_train.reshape(-1,1))
y_test_one_hot=onehot_encoder.fit_transform(y_test.reshape(-1,1))

HOST = '192.168.1.135'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

def send_data(data):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Want to send {np.prod(data.shape)} elements.")
        s.connect((HOST, PORT))

        # data length descriptor as uint32
        length_descriptor = np.uint32(np.prod(data.shape)).tobytes(order='C')

        # payload as C-style float32 byte array
        payload = data.astype(np.float32).tobytes(order='C')
        #print(f"Length {np.uint32(np.prod(data.shape))} bytes, payload {payload}")

        s.sendall(length_descriptor + payload)

np.random.seed(42)
#X = np.concatenate([X_train/255, X_test/255])
#y = np.concatenate([y_train_one_hot, y_test_one_hot])
X = np.random.uniform(0.0,1.0,(600,))
y = np.sin(2*X * math.pi)

send_data(X)
send_data(y)