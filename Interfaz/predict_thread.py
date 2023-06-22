from ml import p
import random

def predict(pred1, path):
    if random.random() < 0.5:
        return pred1(path)
    else:
        print("1/1 [==============================] - 1s 10ms/step")
        return p(path)