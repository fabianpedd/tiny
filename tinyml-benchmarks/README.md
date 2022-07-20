# Generate Self Contained TinyML Benchmarks

This directory contains the core Python script `gen_data_source_files.py` which converts the data from the MLPerf™ Tiny Benchmark into the data format needed for my [TinyML Benchmarks](https://github.com/fabianpedd/tinyml-benchmarks). It creates 4 directories corresponding to the 4 benchmark types:

- [`/aww_data`](./aww_data) Audio wake word - Keyword spotting using a depthwise separable convolutional neural network (DS-CNN)
- [`/ic_data`](./ic_data) Image classification - Small image classification using the Cifar10 dataset and a ResNet model architecture
- [`/toy_data`](./toy_data) ToyCar anomaly detection - Detecting anomalies in machine operating sounds using a deep autoencoder
- [`/vww_data`](./vww_data) Visual wake words - Image classification based on the presence of people using a MobileNet architecture

All generated files are already part of the repository. For ease of use they are **not** ignored. However, should you wish to generate these files yourself, please see below.

## Generate Raw Data
Generating these files in each directory involves a two-step process. First, you need to generate the "raw" data in the corresponding benchmarking directory of the MLPerf™ Tiny Benchmark. For this, please do the following

#### Prerequisite
If you need to run Python scripts you will most likely need to run them inside a virtual environment in which you will need to install the required packages beforehand:

- Change into the directory (benchmark) of interest, e.g. `cd benchmark/training/anomaly_detection` for `anomaly_detection` aka. `toy`
- Create a virtual environment `python3.8 -m venv venv`
- Activate virtual environment `. venv/bin/activate`
- Install required packages `pip install -r requirements.txt`

The last three commands are also summarized inside the `prepare_training_env.sh` scripts which you can simply run.

#### `anomaly_detection` - `toy`
Even though this dataset comes with a benchmark (can be downloaded via `./get_dataset.sh`), we are not actually using it. Instead, we are simply generating random data and recording the output of the autoencoder network as a reference. This completely suffices for our applications and makes things a lot simpler. Thus, all you need to do is run:
- `python gen_tflm_data.py`
This will generate both the data files inside `tflm_dataset/` and the expected output `tflm_labels.csv`.

#### `image_classification` - `ic`
- Downoad the Cifar10 dataset using `./download_cifar10_train_resnet.sh`
- Simply interrupt the training process once the download is done since we already have trained models at hand (`CTRL+C`)
- Then `python gen_tflm_labels.py` to generate the labels `tflm_labels.csv`

#### `keyword_spotting` - `aww`
- Run `python make_bin_files.py --bin_file_path=kws_bin_files --feature_type=mfcc --tfl_file_name=trained_models/kws_ref_model.tflite` to generate the labels
- However, ensure that `test_tfl_on_bin_files = True` in `python make_bin_files.py`
- This will get the data and generate the labels, both inside inside `kws_bin_files/`

#### `visual_wake_words` - `vww`
- Run `download_and_train_vww.sh` to download the data, simply interrupt the training process once the download is done (we already have trained models at hand `CTRL+C`)
- Run `python gen_tflm_samples.py` to generate data and labels, both inside inside `tflm_dataset/`
**Because of its size the `vw_coco2014_96/` is not part of this repo! But you can simply download it with the above commands.** 

## Generate Benchmark Files
Once you have accumulated the "raw" data you need to run the generation script `gen_data_source_files.py`. This will generate the required C source files for the benchmark. UseCopy them over into the corresponding directory inside the [TinyML Benchmarks](https://github.com/fabianpedd/tinyml-benchmarks) repository.

If you only need a subset of the benchmarks simply comment out the call to the corresponding function. If you need to change the amount of tests per benchmark adjust the `NUM_TEST_SAMPLES` variable in each function.

Happy benchmarking!
