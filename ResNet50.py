import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np
import matplotlib.pyplot as plt

model = ResNet50(weights = None)