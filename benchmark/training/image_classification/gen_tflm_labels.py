import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
import pathlib
import os
import re

### Uncomment this to generate the real TFLM lables 
interpreter = tf.lite.Interpreter(model_path="trained_models/pretrainedResnet_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

pathlist = sorted(pathlib.Path("perf_samples").glob('*.bin'))
with open("tflm_labels.csv", "w") as output_file:
    for path in pathlist:
        data = np.fromfile(path, dtype='int8').reshape((1, 32, 32, 3))
        data = (data + 128).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_file.write(f"{os.path.basename(path)},{10},{np.argmax(output_data)}\n")

### Uncomment this to check the real TFLM lables against reference and calculate the acc, should be about 87%
# total_cnt = 0
# correct_cnt = 0
# with open("tflm_labels.csv", "r") as tflm_file:
#     with open("y_labels_sorted.csv", "r") as labels_file:
#         for tflm_line in tflm_file:
#             tflm_label = re.search(",10,[0-9]\n", tflm_line)[0][4:5]
#             true_label = re.search(",10,[0-9]\n", labels_file.readline())[0][4:5]
#             # print(tflm_label, true_label, tflm_label == true_label)
#             # print(tflm_label)
#             if tflm_label == true_label:
#                 correct_cnt += 1
#             total_cnt += 1
#         print(f"Acc: {correct_cnt/total_cnt} after {total_cnt} samples")
