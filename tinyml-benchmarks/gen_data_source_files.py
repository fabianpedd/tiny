#! /usr/bin/env python3

import subprocess
import argparse
import pathlib
import re

################################################################################
################################# AWW ##########################################
################################################################################
def gen_aww():
    NUM_TEST_SAMPLES = 25 # up to 1000

    pathlist = sorted(pathlib.Path("../benchmark/training/keyword_spotting/kws_bin_files").glob('*.bin'))
    pathlib.Path("aww_data/").mkdir(parents=True, exist_ok=True)

    with open("aww_data/aww_input_data.cc", "w") as output:
        output.write("#include \"aww_input_data.h\"\n\n")
        array_name_list = []
        array_name_len_list = []
        for path in pathlist[:NUM_TEST_SAMPLES]:
            ps = subprocess.run(["xxd", "-i", path], stdout=subprocess.PIPE)
            array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
            index = re.search("_[0-9]+_", ps.stdout.decode("utf-8"))[0]
            assert array_data.count("0x") == 490
            array_name = f"aww_input_data_{index[1:-1]}"
            output.write(f"const uint8_t {array_name}[] = {array_data};\n")
            array_name_len = f"{array_name}_len"
            output.write(f"const size_t {array_name_len} = 490;\n\n")
            array_name_list.append(array_name)
            array_name_len_list.append(array_name_len)
        array_name_list = ", ".join(array_name_list)
        output.write(f"const uint8_t* aww_input_data[] = {{{array_name_list}}};\n\n")
        array_name_len_list = ", ".join(array_name_len_list)
        output.write(f"const size_t aww_input_data_len[] = {{{array_name_len_list}}};\n\n")

    with open("aww_data/aww_input_data.h", "w") as output:
        output.write("#ifndef AWW_INPUT_DATA_H \n")
        output.write("#define AWW_INPUT_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write(f"const size_t aww_data_sample_cnt = {NUM_TEST_SAMPLES};\n")
        output.write("extern const uint8_t* aww_input_data[];\n")
        output.write("extern const size_t aww_input_data_len[];\n\n")
        output.write("#endif /* AWW_INPUT_DATA_H */ \n\n")

    with open("aww_data/aww_output_data_ref.cc", "w") as output:
        output.write("#include \"aww_output_data_ref.h\"\n\n")
        with open("../benchmark/training/keyword_spotting/kws_bin_files/tflm_labels.csv") as input:
            data = input.read()
            labels = re.findall(", 12, [0-9]+\n", data)[:NUM_TEST_SAMPLES]
            labels = [x[6:-1] for x in labels]
            labels = ", ".join(labels)
            output.write(f"const uint8_t aww_output_data_ref[] = {{{labels}}};\n\n")

    with open("aww_data/aww_output_data_ref.h", "w") as output:
        output.write("#ifndef AWW_OUTPUT_DATA_REF_H \n")
        output.write("#define AWW_OUTPUT_DATA_REF_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint8_t aww_output_data_ref[];\n\n")
        output.write("#endif /* AWW_OUTPUT_DATA_REF_H */ \n\n")

    with open("aww_data/aww_model_data.cc", "w") as output:
        output.write("#include \"aww_model_data.h\"\n\n")
        ps = subprocess.run(["xxd", "-i", "../benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"], stdout=subprocess.PIPE)
        array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
        len_data = re.search("_len = [0-9]+;", ps.stdout.decode("utf-8"))[0][7:-1]
        assert array_data.count("0x") == int(len_data)
        output.write(f"const uint8_t aww_model_data[] __attribute__((aligned(16))) = {array_data};\n")
        output.write(f"const size_t aww_model_data_size = {len_data};\n\n")

    with open("aww_data/aww_model_data.h", "w") as output:
        output.write("#ifndef AWW_MODEL_DATA_H \n")
        output.write("#define AWW_MODEL_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint8_t aww_model_data[];\n")
        output.write("extern const size_t aww_model_data_size;\n\n")
        output.write("#endif /* AWW_MODEL_DATA_H */ \n\n")

    with open("aww_data/aww_model_settings.cc", "w") as output:
        output.write("#include \"aww_model_settings.h\"\n\n")
        output.write("const char* aww_model_labels[] = {\"down\", \"go\", \"left\", \"no\", \"off\", \"on\", \"right\", \"stop\", \"up\", \"yes\", \"silence\", \"unknown\"};\n")

    with open("aww_data/aww_model_settings.h", "w") as output:
        output.write("#ifndef AWW_MODEL_SETTINGS_H \n")
        output.write("#define AWW_MODEL_SETTINGS_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("const size_t aww_model_label_cnt = 12;\n")
        output.write("extern const char* aww_model_labels[];\n\n")
        output.write("#endif /* AWW_MODEL_SETTINGS_H */ \n\n")

################################################################################
################################# IC ###########################################
################################################################################
def gen_ic():
    NUM_TEST_SAMPLES = 25 # up to 200

    pathlist = sorted(pathlib.Path("../benchmark/training/image_classification/perf_samples").glob('*.bin'))
    pathlib.Path("ic_data/").mkdir(parents=True, exist_ok=True)

    with open("ic_data/ic_input_data.cc", "w") as output:
        output.write("#include \"ic_input_data.h\"\n\n")
        array_name_list = []
        array_name_len_list = []
        for path in pathlist[:NUM_TEST_SAMPLES]:
            ps = subprocess.run(["xxd", "-i", path], stdout=subprocess.PIPE)
            array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
            index = re.search("_[0-9]+_", ps.stdout.decode("utf-8"))[0]
            assert array_data.count("0x") == 3072
            array_name = f"ic_input_data_{index[1:-1]}"
            output.write(f"const uint8_t {array_name}[] = {array_data};\n")
            array_name_len = f"{array_name}_len"
            output.write(f"const size_t {array_name_len} = 3072;\n\n")
            array_name_list.append(array_name)
            array_name_len_list.append(array_name_len)
        array_name_list = ", ".join(array_name_list)
        output.write(f"const uint8_t* ic_input_data[] = {{{array_name_list}}};\n\n")
        array_name_len_list = ", ".join(array_name_len_list)
        output.write(f"const size_t ic_input_data_len[] = {{{array_name_len_list}}};\n\n")

    with open("ic_data/ic_input_data.h", "w") as output:
        output.write("#ifndef IC_INPUT_DATA_H \n")
        output.write("#define IC_INPUT_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write(f"const size_t ic_data_sample_cnt = {NUM_TEST_SAMPLES};\n")
        output.write("extern const uint8_t* ic_input_data[];\n")
        output.write("extern const size_t ic_input_data_len[];\n\n")
        output.write("#endif /* IC_INPUT_DATA_H */ \n\n")

    with open("ic_data/ic_output_data_ref.cc", "w") as output:
        output.write("#include \"ic_output_data_ref.h\"\n\n")
        with open("../benchmark/training/image_classification/tflm_labels.csv") as input:
            data = input.read()
            labels = re.findall(",10,[0-9]+\n", data)[:NUM_TEST_SAMPLES]
            labels = [x[4:-1] for x in labels]
            labels = ", ".join(labels)
            output.write(f"const uint8_t ic_output_data_ref[] = {{{labels}}};\n\n")

    with open("ic_data/ic_output_data_ref.h", "w") as output:
        output.write("#ifndef IC_OUTPUT_DATA_REF_H \n")
        output.write("#define IC_OUTPUT_DATA_REF_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint8_t ic_output_data_ref[];\n\n")
        output.write("#endif /* IC_OUTPUT_DATA_REF_H */ \n\n")

    with open("ic_data/ic_model_data.cc", "w") as output:
        output.write("#include \"ic_model_data.h\"\n\n")
        ps = subprocess.run(["xxd", "-i", "../benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite"], stdout=subprocess.PIPE)
        array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
        len_data = re.search("_len = [0-9]+;", ps.stdout.decode("utf-8"))[0][7:-1]
        assert array_data.count("0x") == int(len_data)
        output.write(f"const uint8_t ic_model_data[] __attribute__((aligned(16))) = {array_data};\n")
        output.write(f"const size_t ic_model_data_size = {len_data};\n\n")

    with open("ic_data/ic_model_data.h", "w") as output:
        output.write("#ifndef IC_MODEL_DATA_H \n")
        output.write("#define IC_MODEL_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint8_t ic_model_data[];\n")
        output.write("extern const size_t ic_model_data_size;\n\n")
        output.write("#endif /* IC_MODEL_DATA_H */ \n\n")

    with open("ic_data/ic_model_settings.cc", "w") as output:
        output.write("#include \"ic_model_settings.h\"\n\n")
        output.write("const char* ic_model_labels[] = {\"airplane\", \"automobile\", \"bird\", \"cat\", \"deer\", \"dog\", \"frog\", \"horse\", \"ship\", \"truck\"};\n")

    with open("ic_data/ic_model_settings.h", "w") as output:
        output.write("#ifndef IC_MODEL_SETTINGS_H \n")
        output.write("#define IC_MODEL_SETTINGS_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("const size_t ic_model_label_cnt = 10;\n")
        output.write("extern const char* ic_model_labels[];\n\n")
        output.write("#endif /* IC_MODEL_SETTINGS_H */ \n\n")

################################################################################
################################# VWW ##########################################
################################################################################
def gen_vww():
    NUM_TEST_SAMPLES = 25 # up to 100

    pathlist = sorted(pathlib.Path("../benchmark/training/visual_wake_words/tflm_dataset").glob('*.bin'))
    pathlib.Path("vww_data/").mkdir(parents=True, exist_ok=True)

    with open("vww_data/vww_input_data.cc", "w") as output:
        output.write("#include \"vww_input_data.h\"\n\n")
        array_name_list = []
        array_name_len_list = []
        for path in pathlist[:NUM_TEST_SAMPLES]:
            ps = subprocess.run(["xxd", "-i", path], stdout=subprocess.PIPE)
            array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
            index = re.search("_[0-9]+_", ps.stdout.decode("utf-8"))[0]
            assert array_data.count("0x") == 27648
            array_name = f"vww_input_data_{index[1:-1]}"
            output.write(f"const uint8_t {array_name}[] = {array_data};\n")
            array_name_len = f"{array_name}_len"
            output.write(f"const size_t {array_name_len} = 27648;\n\n")
            array_name_list.append(array_name)
            array_name_len_list.append(array_name_len)
        array_name_list = ", ".join(array_name_list)
        output.write(f"const uint8_t* vww_input_data[] = {{{array_name_list}}};\n\n")
        array_name_len_list = ", ".join(array_name_len_list)
        output.write(f"const size_t vww_input_data_len[] = {{{array_name_len_list}}};\n\n")

    with open("vww_data/vww_input_data.h", "w") as output:
        output.write("#ifndef VWW_INPUT_DATA_H \n")
        output.write("#define VWW_INPUT_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write(f"const size_t vww_data_sample_cnt = {NUM_TEST_SAMPLES};\n")
        output.write("extern const uint8_t* vww_input_data[];\n")
        output.write("extern const size_t vww_input_data_len[];\n\n")
        output.write("#endif /* VWW_INPUT_DATA_H */ \n\n")

    with open("vww_data/vww_output_data_ref.cc", "w") as output:
        output.write("#include \"vww_output_data_ref.h\"\n\n")
        with open("../benchmark/training/visual_wake_words/tflm_dataset/tflm_labels.csv") as input:
            data = input.read()
            labels = re.findall("[0-9]\n", data)[:NUM_TEST_SAMPLES]
            labels = [x[:-1] for x in labels]
            labels = ", ".join(labels)
            output.write(f"const uint8_t vww_output_data_ref[] = {{{labels}}};\n\n")

    with open("vww_data/vww_output_data_ref.h", "w") as output:
        output.write("#ifndef VWW_OUTPUT_DATA_REF_H \n")
        output.write("#define VWW_OUTPUT_DATA_REF_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint8_t vww_output_data_ref[];\n\n")
        output.write("#endif /* VWW_OUTPUT_DATA_REF_H */ \n\n")

    with open("vww_data/vww_model_data.cc", "w") as output:
        output.write("#include \"vww_model_data.h\"\n\n")
        ps = subprocess.run(["xxd", "-i", "../benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"], stdout=subprocess.PIPE)
        array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
        len_data = re.search("_len = [0-9]+;", ps.stdout.decode("utf-8"))[0][7:-1]
        assert array_data.count("0x") == int(len_data)
        output.write(f"const uint8_t vww_model_data[] __attribute__((aligned(16))) = {array_data};\n")
        output.write(f"const size_t vww_model_data_size = {len_data};\n\n")

    with open("vww_data/vww_model_data.h", "w") as output:
        output.write("#ifndef VWW_MODEL_DATA_H \n")
        output.write("#define VWW_MODEL_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint8_t vww_model_data[];\n")
        output.write("extern const size_t vww_model_data_size;\n\n")
        output.write("#endif /* VWW_MODEL_DATA_H */ \n\n")

    with open("vww_data/vww_model_settings.cc", "w") as output:
        output.write("#include \"vww_model_settings.h\"\n\n")
        output.write("const char* vww_model_labels[] = {\"person\", \"no person\"};\n")

    with open("vww_data/vww_model_settings.h", "w") as output:
        output.write("#ifndef VWW_MODEL_SETTINGS_H \n")
        output.write("#define VWW_MODEL_SETTINGS_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("const size_t vww_model_label_cnt = 2;\n")
        output.write("extern const char* vww_model_labels[];\n\n")
        output.write("#endif /* VWW_MODEL_SETTINGS_H */ \n\n")

################################################################################
################################ TOY ###########################################
################################################################################
def gen_toy():
    NUM_TEST_SAMPLES = 25 # up to 1000

    pathlist = sorted(pathlib.Path("../benchmark/training/anomaly_detection/tflm_dataset").glob('*.bin'))
    pathlib.Path("toy_data/").mkdir(parents=True, exist_ok=True)

    with open("toy_data/toy_input_data.cc", "w") as output:
        output.write("#include \"toy_input_data.h\"\n\n")
        array_name_list = []
        array_name_len_list = []
        for path in pathlist[:NUM_TEST_SAMPLES]:
            ps = subprocess.run(["xxd", "-i", path], stdout=subprocess.PIPE)
            array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
            index = re.search("_[0-9]+_", ps.stdout.decode("utf-8"))[0]
            assert array_data.count("0x") == 640
            array_name = f"toy_input_data_{index[1:-1]}"
            output.write(f"const uint8_t {array_name}[] = {array_data};\n")
            array_name_len = f"{array_name}_len"
            output.write(f"const size_t {array_name_len} = 640;\n\n")
            array_name_list.append(array_name)
            array_name_len_list.append(array_name_len)
        array_name_list = ", ".join(array_name_list)
        output.write(f"const uint8_t* toy_input_data[] = {{{array_name_list}}};\n\n")
        array_name_len_list = ", ".join(array_name_len_list)
        output.write(f"const size_t toy_input_data_len[] = {{{array_name_len_list}}};\n\n")

    with open("toy_data/toy_input_data.h", "w") as output:
        output.write("#ifndef TOY_INPUT_DATA_H \n")
        output.write("#define TOY_INPUT_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write(f"const size_t toy_data_sample_cnt = {NUM_TEST_SAMPLES};\n")
        output.write("extern const uint8_t* toy_input_data[];\n")
        output.write("extern const size_t toy_input_data_len[];\n\n")
        output.write("#endif /* TOY_INPUT_DATA_H */ \n\n")

    with open("toy_data/toy_output_data_ref.cc", "w") as output:
        output.write("#include \"toy_output_data_ref.h\"\n\n")
        with open("../benchmark/training/anomaly_detection/tflm_labels.csv") as input:
            data = input.read()
            labels = re.findall("[-+]?(?:\d*\.\d+|\d+)\n", data)[:NUM_TEST_SAMPLES]
            labels = [x[:-1] for x in labels]
            labels = ", ".join(labels)
            output.write(f"const uint32_t toy_output_data_ref[] = {{{labels}}};\n\n")

    with open("toy_data/toy_output_data_ref.h", "w") as output:
        output.write("#ifndef TOY_OUTPUT_DATA_REF_H \n")
        output.write("#define TOY_OUTPUT_DATA_REF_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint32_t toy_output_data_ref[];\n\n")
        output.write("#endif /* TOY_OUTPUT_DATA_REF_H */ \n\n")

    with open("toy_data/toy_model_data.cc", "w") as output:
        output.write("#include \"toy_model_data.h\"\n\n")
        ps = subprocess.run(["xxd", "-i", "../benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite"], stdout=subprocess.PIPE)
        array_data = re.search("{[\x00-\x7F]+};", ps.stdout.decode("utf-8"))[0][:-1]
        len_data = re.search("_len = [0-9]+;", ps.stdout.decode("utf-8"))[0][7:-1]
        assert array_data.count("0x") == int(len_data)
        output.write(f"const uint8_t toy_model_data[] __attribute__((aligned(16))) = {array_data};\n")
        output.write(f"const size_t toy_model_data_size = {len_data};\n\n")

    with open("toy_data/toy_model_data.h", "w") as output:
        output.write("#ifndef TOY_MODEL_DATA_H \n")
        output.write("#define TOY_MODEL_DATA_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("extern const uint8_t toy_model_data[];\n")
        output.write("extern const size_t toy_model_data_size;\n\n")
        output.write("#endif /* TOY_MODEL_DATA_H */ \n\n")

    with open("toy_data/toy_model_settings.cc", "w") as output:
        output.write("#include \"toy_model_settings.h\"\n\n")
        output.write("/* Nothing here. */ \n")

    with open("toy_data/toy_model_settings.h", "w") as output:
        output.write("#ifndef TOY_MODEL_SETTINGS_H \n")
        output.write("#define TOY_MODEL_SETTINGS_H \n\n")
        output.write("#include <stdint.h>\n")
        output.write("#include <stddef.h>\n\n")
        output.write("/* There are no labels for the toycar (aka. anomaly detection, aka. autoencoder) model. */\n\n")
        output.write("#endif /* TOY_MODEL_SETTINGS_H */ \n\n")

def main():
    gen_aww()
    gen_ic()
    gen_vww()
    gen_toy()

if __name__ == "__main__":
    main()
