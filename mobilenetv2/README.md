# Getting Started with MobilenetV2

Start by cloning this repo:
* git clone https://github.com/latentai/model-zoo-models.git
* cd mobilenetv2

The following commands should "just work":


# Download pretrained model on Open Images 10 Classes
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes

# Download pretrained imagenet model
./dev_docker_run leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet

# Download dataset for Transfer Learning training

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id train

./dev_docker_run leip zoo download --dataset_id open-images-10-classes --variant_id eval

# Train

(Set --epochs and --batch_size to 1 for a quick training run.)

./dev_docker_run ./train.py --dataset_path workspace/datasets/open-images-10-classes/train/  --eval_dataset_path workspace/datasets/open-images-10-classes/eval/ --epochs 150

# Evaluate a trained model

./dev_docker_run ./eval.py --dataset_path workspace/datasets/open-images-10-classes/eval/ --input_model_path trained_model/model.h5

# Demo

This runs inference on a single image.
./dev_docker_run ./demo.py --input_model_path trained_model/model.h5 --image_file test_images/dog.jpg

# LEIP SDK Post-Training-Quantization Commands on Pretrained Models

Open Image 10 Classes Commands
# Preparation
leip zoo download --model_id mobilenetv2 --variant_id keras-open-images-10-classes
leip zoo download --dataset_id open-images-10-classes --variant_id eval
rm -rf mobilenetv2-oi
mkdir mobilenetv2-oi
mkdir mobilenetv2-oi/baselineFp32Results
# CMD#1 Baseline FP32 TF
leip evaluate --output_path mobilenetv2-oi/baselineFp32Results --framework tf --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
# LEIP Compress ASYMMETRIC
leip compress --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --quantizer ASYMMETRIC --bits 8 --output_path mobilenetv2-oi/checkpointCompressed/
# LEIP Compress POWER_OF_TWO (POW2)
leip compress --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --quantizer POWER_OF_TWO --bits 8 --output_path mobilenetv2-oi/checkpointCompressedPow2/
# CMD#2 LEIP FP32 TF
leip evaluate --output_path mobilenetv2-oi/checkpointCompressed/ --framework tf --input_path mobilenetv2-oi/checkpointCompressed/model_save/ --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
# CMD#3 Baseline INT8 TVM
rm -rf mobilenetv2-oi/compiled_tvm_int8
mkdir mobilenetv2-oi/compiled_tvm_int8
leip compile --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --output_path mobilenetv2-oi/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-oi/compiled_tvm_int8/ --framework lre --input_types=uint8 --input_path mobilenetv2-oi/compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
# CMD#4 Baseline FP32 TVM
rm -rf mobilenetv2-oi/compiled_tvm_fp32
mkdir mobilenetv2-oi/compiled_tvm_fp32
leip compile --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --output_path mobilenetv2-oi/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-oi/compiled_tvm_fp32/ --framework lre --input_types=float32 --input_path mobilenetv2-oi/compiled_tvm_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
# CMD#5 LEIP INT8 TVM
rm -rf mobilenetv2-oi/leip_compiled_tvm_int8
mkdir mobilenetv2-oi/leip_compiled_tvm_int8
leip compile --input_path mobilenetv2-oi/checkpointCompressed/model_save/ --output_path mobilenetv2-oi/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-oi/leip_compiled_tvm_int8 --framework lre --input_types=uint8 --input_path mobilenetv2-oi/leip_compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
# CMD#6 LEIP FP32 TVM
rm -rf mobilenetv2-oi/leip_compiled_tvm_fp32
mkdir mobilenetv2-oi/leip_compiled_tvm_fp32
leip compile --input_path mobilenetv2-oi/checkpointCompressed/model_save/ --output_path mobilenetv2-oi/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-oi/leip_compiled_tvm_fp32 --framework lre --input_types=float32 --input_path mobilenetv2-oi/leip_compiled_tvm_fp32/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
# CMD#7 LEIP-POW2 INT8 TVM
rm -rf mobilenetv2-oi/leip_compiled_tvm_int8_pow2
mkdir mobilenetv2-oi/leip_compiled_tvm_int8_pow2
leip compile --input_path mobilenetv2-oi/checkpointCompressedPow2/model_save/ --output_path mobilenetv2-oi/leip_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-oi/leip_compiled_tvm_int8_pow2 --framework lre --input_types=uint8 --input_path mobilenetv2-oi/leip_compiled_tvm_int8/bin --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt
# CMD#8 TfLite Asymmetric INT8 TF
rm -rf mobilenetv2-oi/tfliteOutput
mkdir mobilenetv2-oi/tfliteOutput
leip convert --input_path workspace/models/mobilenetv2/keras-open-images-10-classes --framework tflite --output_path mobilenetv2-oi/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared-workdir/workspace/datasets/open-images-10-classes/eval/Apple/06e47f3aa0036947.jpg
leip evaluate --output_path mobilenetv2-oi/tfliteOutput --framework tflite --input_types=uint8 --input_path mobilenetv2-oi/tfliteOutput/model_save/inference_model.cast.tflite --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt --preprocessor ''
# CMD#9 TfLite Asymmetric INT8 TVM
leip compile --input_path mobilenetv2-oi/tfliteOutput/model_save/inference_model.cast.tflite --output_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path mobilenetv2-oi/tfliteOutput/model_save/binuint8 --test_path workspace/datasets/open-images-10-classes/eval/index.txt --class_names workspace/models/mobilenetv2/keras-open-images-10-classes/class_names.txt --preprocessor ''


Imagenet Commands
# Preparation
leip zoo download --model_id mobilenetv2 --variant_id keras-imagenet
rm -rf mobilenetv2-imagenet
mkdir mobilenetv2-imagenet
mkdir mobilenetv2-imagenet/baselineFp32Results
# CMD#10 Baseline FP32 TF
leip evaluate --output_path mobilenetv2-imagenet/baselineFp32Results --framework tf --input_path workspace/models/mobilenetv2/keras-imagenet --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# LEIP Compress ASYMMETRIC
leip compress --input_path workspace/models/mobilenetv2/keras-imagenet --quantizer ASYMMETRIC --bits 8 --output_path mobilenetv2-imagenet/checkpointCompressed/
# LEIP Compress POWER_OF_TWO (POW2)
leip compress --input_path workspace/models/mobilenetv2/keras-imagenet --quantizer POWER_OF_TWO --bits 8 --output_path mobilenetv2-imagenet/checkpointCompressedPow2/
# CMD#11 LEIP FP32 TF
leip evaluate --output_path mobilenetv2-imagenet/checkpointCompressed/ --framework tf --input_path mobilenetv2-imagenet/checkpointCompressed/model_save/ --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#12 Baseline INT8 TVM
rm -rf mobilenetv2-imagenet/compiled_tvm_int8
mkdir mobilenetv2-imagenet/compiled_tvm_int8
leip compile --input_path workspace/models/mobilenetv2/keras-imagenet --output_path mobilenetv2-imagenet/compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-imagenet/compiled_tvm_int8/ --framework lre --input_types=uint8 --input_path mobilenetv2-imagenet/compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#13 Baseline FP32 TVM
rm -rf mobilenetv2-imagenet/compiled_tvm_fp32
mkdir mobilenetv2-imagenet/compiled_tvm_fp32
leip compile --input_path workspace/models/mobilenetv2/keras-imagenet --output_path mobilenetv2-imagenet/compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-imagenet/compiled_tvm_fp32/ --framework lre --input_types=float32 --input_path mobilenetv2-imagenet/compiled_tvm_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#14 LEIP INT8 TVM
rm -rf mobilenetv2-imagenet/leip_compiled_tvm_int8
mkdir mobilenetv2-imagenet/leip_compiled_tvm_int8
leip compile --input_path mobilenetv2-imagenet/checkpointCompressed/model_save/ --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8 --framework lre --input_types=uint8 --input_path mobilenetv2-imagenet/leip_compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#15 LEIP FP32 TVM
rm -rf mobilenetv2-imagenet/leip_compiled_tvm_fp32
mkdir mobilenetv2-imagenet/leip_compiled_tvm_fp32
leip compile --input_path mobilenetv2-imagenet/checkpointCompressed/model_save/ --output_path mobilenetv2-imagenet/leip_compiled_tvm_fp32/bin --input_types=float32 --data_type=float32
leip evaluate --output_path mobilenetv2-imagenet/leip_compiled_tvm_fp32 --framework lre --input_types=float32 --input_path mobilenetv2-imagenet/leip_compiled_tvm_fp32/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#16 LEIP-POW2 INT8 TVM
rm -rf mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2
mkdir mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2
leip compile --input_path mobilenetv2-imagenet/checkpointCompressedPow2/model_save/ --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2/bin --input_types=uint8 --data_type=int8
leip evaluate --output_path mobilenetv2-imagenet/leip_compiled_tvm_int8_pow2 --framework lre --input_types=uint8 --input_path mobilenetv2-imagenet/leip_compiled_tvm_int8/bin --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom
# CMD#17 TfLite Asymmetric INT8 TF
rm -rf mobilenetv2-imagenet/tfliteOutput
mkdir mobilenetv2-imagenet/tfliteOutput
leip convert --input_path workspace/models/mobilenetv2/keras-imagenet --framework tflite --output_path mobilenetv2-imagenet/tfliteOutput --data_type int8 --policy TfLite --rep_dataset /shared/data/sample-models/resources/images/imagenet_images/preprocessed/ILSVRC2012_val_00000001.JPEG
leip evaluate --output_path mobilenetv2-imagenet/tfliteOutput --framework tflite --input_types=uint8 --input_path mobilenetv2-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom --preprocessor ''
# CMD#18 TfLite Asymmetric INT8 TVM
leip compile --input_path mobilenetv2-imagenet/tfliteOutput/model_save/inference_model.cast.tflite --output_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --input_types=uint8
leip evaluate --output_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --framework lre --input_types=uint8 --input_path mobilenetv2-imagenet/tfliteOutput/model_save/binuint8 --test_path /shared/data/sample-models/resources/data/imagenet/testsets/testset_1000_images.preprocessed.1000.txt --class_names workspace/models/mobilenetv2/keras-imagenet/class_names.txt --task=classifier --dataset=custom --preprocessor ''
