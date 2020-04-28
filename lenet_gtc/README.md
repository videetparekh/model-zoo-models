# MNIST dataset

This code uses the built-in data library MNIST.
It is downloaded automatically at the first run,, and is used afterwards.
Internal Tensorflow libraries are used for that 
`from tensorflow.keras.datasets import mnist`

# Train

Training of the model is performed "from scratch". Since the model is simple and the dataset is relatively
small, the whole training might take around 30 seconds for one epoch. 
10 epochs is usually enough to get a decent result.


Basic training command:
`python3 train.py --basedirectory ***`
which is equivalent to the following default parameters:
`python3 train.py --basedirectory *** --batch_size 64 --learning_rate 0.0002 --lambda_bit_loss 1e-5 --lambda_distillation_loss 0.01 --lambda_regularization 0.01`

ADAM optimization is used.

Important moment to take into account is `lambda_bit_loss` and `lambda_distillation_loss`. These two arguments control the quantization process. Their values are chosen so that a model accurate enough and with low number of bits is created! 
Increasing `lambda_distillation_loss` loss increases accuracy.
Increasing `lambda_bit_loss` decreases average number of bits used.
The regularization term, determined by `lambda_distillation_loss` pulls all the weights down towards zero. Very often it helps the GTC model converge to a better local solution. It is not meaningful for the LeNet system and is presented here as a demonstration of system capabilities.

If the values of the lambdas are too high, the system can not attain high accuracy, so it is not recommended to increase them by more then 20 times.

Expected result after 5 epochs:
`Test results after epoches 4`

`('lp_accuracy', 'dense_2') 0.94`

`('hp_accuracy', 'dense_2') 0.97`

`('total_loss', 'total_loss') 0.11`

`bit_loss 130.15`

`('distillation_loss', 'dense_2') 0.20`

`('hp_cross_entropy', 'dense_2') 0.08`

`regularization_term 202.27`

The rseults are located in `./lenet_on_mnist_adam_weight_decay_0.0002_lam_bl_1e-05_lam_dl_0.01`
which will be created during the run. Main directories there:
* `training_model_final` - contains the final trained HP model
* `int_model_final`      - contains the final trained LP model

These two models can be further taken through the leip pipeline and the evaluations and the compilations scripts, explained next.

For the sequel perform `cd lenet_on_mnist_adam_weight_decay_0.0002_lam_bl_1e-05_lam_dl_0.01`

# Compile

Compilinig the models:
* `leip compile --input_path ./training_model_final/ --output_path ./HPcompiled`
will create HO compiled model in directory HPcompiled.
* `leip compile --input_path ./int_model_final/ --output_path ./LPcompiled`
creates LP model in LPcompiled folder.

You can check that the outputs of these two models work correctly using 
```
leip run --input_path /home/model-zoo-models/lenet_gtc/lenet_on_mnist_adam_weight_decay_0.0002_lam_bl_1e-05_lam_dl_0.01/HPcompiled/ --test_path $MNISTDIR/images/zero.jpg --class_names $MNISTDIR/class_names.txt --input_names Placeholder --output_names Softmax --input_shapes 1,28,28
```
where the variable MNISTDIR consists of the full path to ../mnist_directory (it is included together with the repository). There are a few more files that can be replaced for zero.jpg.

Similarly for the LP model one can check that (notice `HPcompiled` is repolaced with `LPcompiled` here)
```
leip run --input_path /home/model-zoo-models/lenet_gtc/lenet_on_mnist_adam_weight_decay_0.0002_lam_bl_1e-05_lam_dl_0.01/LPcompiled/ --test_path $MNISTDIR/images/zero.jpg --class_names $MNISTDIR/class_names.txt --input_names Placeholder --output_names Softmax --input_shapes 1,28,28.
```

# Compression of the model
Compressing the HP model with 5 bits for each layer
```
leip compress --input_path training_model_final/ --bits 5 --output_path HPcompressed
```
Similarly, one can compress the LP model:
```
leip compress --input_path int_model_final/ --bits 5 --output_path LPcompressed
```

# Evaluation of the models
The script `create_data.py` creates data for evaluation. The data is already created in the folder 
`mnist_examples`.

Checking the compressed models for HPcompressed:
```
leip evaluate --framework tf2 --input_path ./HPcompressed/model_save/ --test_path ../mnist_examples/index.txt --class_names ../mnist_examples/class_names.txt --task=classifier --dataset=custom --input_names Placeholder --output_names Softmax --input_shapes 1,28,28
```
Now it is possible to check that each one of the earlier created models works. Instead of the directory
`HPcompressed/model_save/` insert `LPcompressed/model_save/`, `training_model_final/` or `int_model_final/`.

We can also check that the compiled models work as follows (notice the change in framework `tf2->tvm` and the input_path directory):
```
leip evaluate --framework tvm --input_path ./HPcompiled/ --test_path ../mnist_examples/index.txt --class_names ../mnist_examples/class_names.txt --task=classifier --dataset=custom --input_names Placeholder --output_names Softmax --input_shapes 1,28,28
```
or with `LPcompiled` instead of `HPcompiled`.

