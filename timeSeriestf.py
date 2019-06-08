from sklearn.cross_validation import train_test_split
import numpy as np
from matplotlib import pyplot as plt
 
import tensorflow as tf
 
from tensorflow.python.platform import tf_logging as logging
 
logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)
 
 
def generate_time_series(datalen):
    freq1 = 0.2
    freq2 = 0.15
    noise = [np.random.random() * 0.1 for i in range(datalen)]
    x1 = np.sin(np.arange(0, datalen) * freq1) + noise
    x2 = np.sin(np.arange(0, datalen) * freq2) + noise 
    x = x1 + x2
    return x.astype(np.float32)
 
 
DATA_SEQ_LEN = 24000
 
data = generate_time_series(DATA_SEQ_LEN)