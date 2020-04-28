from tensorflow.keras.datasets import mnist
import random
import imageio

(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(1):
    if y_test[i]==0:
        imageio.imwrite('images/zero/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==1:
        imageio.imwrite('images/one/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==2:
        imageio.imwrite('images/two/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==3:
        imageio.imwrite('images/three/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==4:
        imageio.imwrite('images/four/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==5:
        imageio.imwrite('images/five/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==6:
        imageio.imwrite('images/six/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==7:
        imageio.imwrite('images/seven/'+round(random.random()*10000)+'.jpg',x_test[i])
    elif y_test[i]==8:
        imageio.imwrite('images/eight/'+round(random.random()*10000)+'.jpg',x_test[i])
    else:
        imageio.imwrite('images/nine/'+round(random.random()*10000)+'.jpg',x_test[i])
    

