import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
import pathlib
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

### For the autoencoder we just generate random input and random output data.
### This data can then be compared later on in order to test the inference code
### for correctness.

np.random.seed(0)

### How many samples to generate
NUM_SAMPLES = 100

interpreter = tf.lite.Interpreter(model_path="trained_models/ad01_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

pathlib.Path("tflm_dataset/").mkdir(parents=True, exist_ok=True)

with open("tflm_labels.csv", "w") as output_file:
    for idx in range(NUM_SAMPLES):
        input_data = (np.random.rand(1, 640) * 255).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        score = np.mean(np.square(input_data.astype(np.int32) - output_data.astype(np.int32)), axis=1).astype(np.int32)
        input_data.astype('int8').tofile(f"tflm_dataset/{idx:04d}.bin")
        output_file.write(f"{score[0]}\n")
