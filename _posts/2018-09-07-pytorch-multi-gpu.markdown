#caffe入门学习
 作者：tequilaro
本文主要讲解caffe的整个使用流程，适用于初级入门，因为自己一直在做这一方面的学习，希望可以通过本篇文章给大家一些指点，最主要是要理清自己的思路，以便更好地学习。如果有比我更好的解决方法希望大家可以多多给我提出建议。

#####一、总流程
一般的流程如下所示
 1. 数据格式处理，一般caffe处理的均为图片数据，然后我们需要将图片数据以及相关标签打包在一起，实现caffe的调用。
 2.编写网络结构文件，通常后缀格式为.prototxt。一般caffe中都会自带手写字体识别这一程序，具体目录在caffe/examples/mnist/中，其中相关的网络结构文件为lenet_train_test.prototxt。这个就是手写字体的网络结构文件。
 3.网络配置文件，一般我们都取名为：solver.prototxt，这个文件中一般包含梯度下降参数，学习率，迭代次数等。具体的参数配置可以见这篇文章 [caffe中的一些参数介绍](http://blog.csdn.net/cyh_24/article/details/51537709)
 4. 编写脚本文件实现网络的训练， 接着我们调用caffe可执行文件进行训练就可以了。
 这个是caffe的一般流程。但是每个具体的项目其实并不相同所以我们需要举一反三，通过caffe中提供的例子来应用到我们自己的实际项目中。
 


在我项目中的具体应用，我在训练的时候遇到了许多问题，下面就和大家一一说来，说不定可以给大家的日常学习提供一些帮助。

因为之前接触的caffe都是处理的是图片，然后在通过自caffe已经编写好的create_mnist.sh的脚本程序生成lmdb的数据，在将标签与图片一一对应放入到写好的网络中，从而可以训练网络。
但是从我上的上一篇文章中可以得知我处理的数据均不是图片数据，所以需要考虑新的方法来解决数据。

#####博主主要采用了两种方式进行数据处理。

 一.第一种方法是将二维矩阵转变为图片。因为观察自己的数据均在0-255之间，可以转变为RGB色彩模式，所以预期是期望将每个5*3的二维矩阵转变为一个以RGB格式存储的图片。之后再利用之前的流程进行图片数据的处理，针对不同的图片在打不同的标签。但是通过尝试之后，发现将二维矩阵转变为RGB方式存储后的得到的图片，通过已经搭建好的网络进行训练时，accuracy一直为1，或者为0。loss也高达88.3365。
 不论如何调整学习率都不能得到理想的结果。而且通过我们肉眼也发现将我们之前的二维矩阵转变为图片也的确没有特征可言。下面是我的几个处理好的二维矩阵，与之相对应的是通过程序将其转变为RGB三通道的图片。

```
plt.imshow(A[i])
#是利用这行代码将二维矩阵数据转换为RGB格式存储的图片
```

 二.第二种方法就是发现caffe处理数据的格式有lmdb，HDF5等格式，所以思考是否可以将我们的二维矩阵直接存储为lmdb格式的数据，不使用caffe自带生成数据的文件，而是自己编写。最终参考这篇文章得到了问题的解决方案。
[Creating an LMDB database in Python](http://deepdish.io/2015/04/28/creating-lmdb-in-python/)
最终这种方法得到了较为理想的结果。

#####接下来详细讲述这种方法的步骤流程，可能下面出现的代码比较繁琐，但是博主还在不断更新，努力学习中，望见谅。
大致流程为：
1.进行数据处理，将数据处理为lmdb格式 
2.运行create_lmdb.sh文件建立网络所需的训练数据
3.编写训练网络文件
4.调整编写网络的参数，学习率以及卷积核大小等
5.训练已经编写的网络，得到其准确率和损失率。
######1.lmdb数据格式生成
先来介绍自己训练集数据的生成
```
# coding=utf-8
import numpy as np
import lmdb
import sys
import os
import random

#这里主要是寻找到caffe的路径
caffe_root = '/home/ly/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
os.environ['GLOG_minloglevel'] = '3'
caffe_root = os.path.expanduser('~/caffe/')  # change with your install location
sys.path.insert(0, os.path.join(caffe_root, 'python'))
sys.path.insert(0, os.path.join(caffe_root, 'python/caffe/proto'))
sys.path.insert(0, os.path.join(caffe_root, 'python/caffe/tripletloss'))
import caffe

N_test = 298

# Let's pretend this is interesting data
X = np.zeros((N_test, 1, 5, 3), dtype=np.float)
y = np.zeros(N_test, dtype=np.int64)
map_size = X.nbytes * 10
#这里是处理我的训练集数据
A_1 = [0] * 3
A_2 = [A_1] * 5
A = [A_2] * 596
a = np.loadtxt("data.txt")
# print(a)
for row in range(0, 596):
	col1 = row
	col2 = row + 5
	A[row] = a[col1:col2]
A = np.array(A)
x_test = A[1::2]
#这里是处理我的训练集标签
b = np.loadtxt("label.txt")
y_test = b[1::2]

for i in range(N_test):
	X[i][0] = x_test[i]
for i in range(N_test):
	y[i] = y_test[i]
env = lmdb.open('test_lmdb', map_size=map_size)

with env.begin(write=True) as txn:
	# txn is a Transaction object
	for i in range(N_test):
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = X.shape[1]
		datum.height = X.shape[2]
		datum.width = X.shape[3]
		datum.data = X[i].tobytes()
		datum.label = int(y[i])
		str_id = '{:08}'.format(i)
		
		txn.put(str_id.encode('ascii'), datum.SerializeToString())

```
通过运行train_data.py文件生成train_lmdb文件。验证集的数据同理。
######2.编写网络文件
通过学习对原网络进行了调整以及修改，而且增加了mirror:true目的是为了对数据进行预处理，对数据进行镜像处理。
######3.调整网络参数
######4.训练网络
最终通过训练网络，在迭代了1000次之后得到的准确率以及损失率，得到了相对满意的结果。
#####本篇文章主要涉及到的是将二维矩阵转变为图片从中提取特征或者直接存储为lmdb格式从中提取特征两种方法的介绍，如果本篇文章中有什么不正确之处也欢迎大家指出，共同学习，谢谢。



-------------------
