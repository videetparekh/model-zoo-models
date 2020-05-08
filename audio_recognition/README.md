# Getting Started with Audio Recognition

Start by cloning this repo:
* git clone https://github.com/latentai/model-zoo-models.git
* cd audio_recognition

The following commands should "just work":

# Download dataset

./dev_docker_run leip zoo download --dataset_id google-speech-commands --variant_id v0.02

# Train the model

--how_many_training_steps 10000,10000 means 10000 high learning rate steps followed by 10000 low learning rate steps.

Set --how_many_training_steps 1,1 for a fast training run.

`./dev_docker_run python train.py --how_many_training_steps 10000,10000 --eval_step_interval 2000 --data_dir datasets/google-speech-commands/v0.02/speech_commands/ --train_dir train_data --wanted_words backward,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero`

Once model trained `train_data` directory will be created. It will contain tensorflow summaries and checkpoint of trained model.

# Evaluate the model

In order to evaluate the model on test set run following command:
(Adjust the file name passed to --checkpoint if needed)

`./dev_docker_run python eval.py --checkpoint train_data/conv.ckpt-20000 --data_dir datasets/google-speech-commands/v0.02/speech_commands/ --train_dir train_data --wanted_words backward,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero`

# Demo

To make a prediction on wav file run following command:
(Adjust the file name passed to --checkpoint if needed)

`./dev_docker_run python demo.py --checkpoint train_data/conv.ckpt-20000 --data_dir datasets/google-speech-commands/v0.02/speech_commands/ --train_dir train_data --wanted_words backward,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero --wav datasets/google-speech-commands/v0.02/speech_commands/cat/030ec18b_nohash_1.wav`

This command will output the prediction of word "cat".

# Download pretrained checkpoint

`./dev_docker_run leip zoo download --model_id audio-recognition --variant_id tf-baseline`

# Evaluate pretrained checkpoint

`./dev_docker_run python eval.py --checkpoint models/audio-recognition/tf-baseline/pretrained_tf_checkpoint/conv.ckpt-35000 --data_dir datasets/google-speech-commands/v0.02/speech_commands/ --train_dir train_data --wanted_words backward,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero`

# LEIP part

## Compress tensorflow checkpoint

***Asymetric***

`rm -rf checkpoint_compressed_asym && leip compress --input_path train_data/ --quantizer ASYMMETRIC --bits 8 --output_path checkpoint_compressed_asym/`

`./dev_docker_run python eval.py --checkpoint checkpoint_compressed_asym/model_save/new_model --data_dir datasets/google-speech-commands/v0.02/speech_commands/ --train_dir train_data --wanted_words backward,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero`

***Power of two***

`rm -rf checkpoint_compressed_pow2/ && leip compress --input_path train_data/ --quantizer POWER_OF_TWO --bits 8 --output_path checkpoint_compressed_pow2/`

`./dev_docker_run python eval.py --checkpoint checkpoint_compressed_pow2/model_save/new_nodel --data_dir datasets/google-speech-commands/v0.02/speech_commands/ --train_dir train_data --wanted_words backward,bed,bird,cat,dog,down,eight,five,follow,forward,four,go,happy,house,learn,left,marvin,nine,no,off,on,one,right,seven,sheila,six,stop,three,tree,two,up,visual,wow,yes,zero`

## Compile checkpoints into int8

`rm -rf compiled_tf_tvm_int8 && mkdir compiled_tf_tvm_int8 && leip compile --input_path train_data/ --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_tf_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_asym_tvm_int8 && mkdir compiled_asym_tf_tvm_int8 && leip compile --input_path checkpoint_compressed_asym/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_asym_tf_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_pow2_tvm_int8 && mkdir compiled_pow2_tf_tvm_int8 && leip compile --input_path checkpoint_compressed_pow2/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_pow2_tf_tvm_int8/bin --input_types=uint8 --data_type=int8 --input_names fingerprint_input,dropout_prob --output_names add_2`

## Compile tensorflow checkpoint into fp32

`rm -rf compiled_tf_tvm_fp32 && mkdir compiled_tf_tvm_fp32 && leip compile --input_path train_data/ --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_tf_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_asym_tvm_fp32 && mkdir compiled_asym_tvm_fp32 && leip compile --input_path checkpoint_compressed_asym/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_asym_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names fingerprint_input,dropout_prob --output_names add_2`

`rm -rf compiled_pow2_tvm_fp32 && mkdir compiled_pow2_tvm_fp32 && leip compile --input_path checkpoint_compressed_pow2/model_save --input_shapes "1, 224, 224, 3"-"1,1" --output_path compiled_pow2_tvm_fp32/bin --input_types=float32 --data_type=float32 --input_names fingerprint_input,dropout_prob --output_names add_2`
