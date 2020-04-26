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


