import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
import pathlib
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

### How many samples to generate
NUM_SAMPLES = 100

pathlib.Path("tflm_dataset/").mkdir(parents=True, exist_ok=True)

non_person_dir = os.path.join(os.getcwd(), "vw_coco2014_96/non_person")
person_dir = os.path.join(os.getcwd(), "vw_coco2014_96/person")

non_person_files = sorted(pathlib.Path(non_person_dir).glob('*.jpg'))
person_files = sorted(pathlib.Path(person_dir).glob('*.jpg'))

all_files = [x for pair in zip(non_person_files, person_files) for x in pair]

ground_truth_labels = np.tile([0, 1], len(all_files))

### Choose either float or quantized int8 model (NEED TO CHANGE ONE LINE IN FOR LOOP AS WELL!!!!)
### The accuracy should be about 85% for both the float and the int8 model
interpreter = tf.lite.Interpreter(model_path="trained_models/vww_96_int8.tflite")
# interpreter = tf.lite.Interpreter(model_path="trained_models/vww_96_float.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

total_cnt = 0
correct_cnt = 0

with open("tflm_dataset/tflm_labels.csv", "w") as f:
    for idx, file in enumerate(all_files[:NUM_SAMPLES]):
        img = tf.keras.preprocessing.image.load_img(file, color_mode='rgb').resize((96, 96))
        data = tf.keras.preprocessing.image.img_to_array(img)
        # data = data.reshape(1, 96, 96, 3) / 255. # for float model
        data = (data.reshape(1, 96, 96, 3) + 128).astype(np.int8) # for int8 model

        ### Should show non_person and person images interleaved
        # imgplot = plt.imshow(data[0])
        # plt.show()

        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        tflm_label = np.argmax(output_data)
        if tflm_label == ground_truth_labels[idx]:
            correct_cnt += 1
        total_cnt += 1

        data.astype('int8').tofile(f"tflm_dataset/{idx:04d}.bin")
        f.write(f"{tflm_label}\n")

print(f"Acc: {correct_cnt/total_cnt} after {total_cnt} samples")
