from tensorflow.keras.datasets import mnist
import random
import imageio

(x_train, y_train), (x_test, y_test) = mnist.load_data()
file_object = open('index.txt', 'a')
print(len(y_test))
print((len(y_test) + 3) % len(y_test))
for j in range(500):
    i=(j*1234) % len(y_test)
    if y_test[i]==0:
        name='images/zero/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 0')
    elif y_test[i]==1:
        name = 'images/one/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 1')
    elif y_test[i]==2:
        name = 'images/two/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 2')
    elif y_test[i]==3:
        name = 'images/three/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 3')
    elif y_test[i]==4:
        name = 'images/four/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 4')
    elif y_test[i]==5:
        name = 'images/five/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 5')
    elif y_test[i]==6:
        name = 'images/six/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 6')
    elif y_test[i]==7:
        name = 'images/seven/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 7')
    elif y_test[i]==8:
        name = 'images/eight/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 8')
    else:
        name = 'images/nine/'+str(round(random.random()*10000))+'.jpg'
        imageio.imwrite(name,x_test[i])
        file_object.write(name + ' 9')
    file_object.write('\n')

file_object.close()

