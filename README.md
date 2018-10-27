# miniLearn
A mini neural network framework,but currently only supported Convolutional neural network.<br>
The rest is still under development.<br>
<br>
<br>
It is very easy to use.<br>
```cpp
#include "miniLearn.h"
```
<br>
<br>
This has a simple mnist training.<br>
```
#include "struct/CSV.h"
```
<br>
Usage:<br>
<br>
####train<br>
```cpp
  mnist_train mnist;
  
  double *b = new double[10 * 200]{ 0 };
	double *c = new double[784 * 200];

	mnist.getbatch(b, c, 200);                                                       //Read 200 training samples      

	miniLearn mLearn;
	Graph& graph = mLearn.CreateGraph();                                             //Generate an operation diagram

	Tensor y_ = mLearn.Constant(b, 10 * 200, { 10,1,1,1 }, 200);                     //Initialize constants using b

	Tensor x_image = mLearn.Constant(c, 784 * 200, { 28,28,1,1 }, 200);
	Tensor w_conv1 = mLearn.Variable(0, 0.1, { 5,5,1,32 });                          //Initialize variable are normally distributed
	Tensor b_conv1 = mLearn.Variable(0.1, { 1,1,32,1 });                             //Initialize variable are all 0.1
	Tensor h_conv1 = mLearn.Relu(mLearn.add(mLearn.Convolution(x_image, w_conv1, { 1,1,1,1 }, SAME), b_conv1));   //Convolution and activation
	Tensor h_pool1 = mLearn.MaxPool(h_conv1, { 2,2,1,1 }, { 2,2,1,1 }, SAME);        //Pool

	Tensor w_conv2 = mLearn.Variable(0, 0.1, { 5,5,32,64 });
	Tensor b_conv2 = mLearn.Variable(0.1, { 1,1,64,1 });
	Tensor h_conv2 = mLearn.Relu(mLearn.add(mLearn.Convolution(h_pool1, w_conv2, { 1,1,1,1 }, SAME), b_conv2));
	Tensor h_pool2 = mLearn.MaxPool(h_conv2, { 2,2,1,1 }, { 2,2,1,1 }, SAME);

	Tensor w_fc1 = mLearn.Variable(0, 0.1, { 1024,49 * 64,1,1 });
	Tensor b_fc1 = mLearn.Variable(0.1, { 1024,1,1,1 });
	Tensor h_pool2_flat = mLearn.reshape(h_pool2, { 49 * 64,1,1,1 });                 //Reset shape
	Tensor h_fc1 = mLearn.Relu(mLearn.add(mLearn.matmul(h_pool2_flat, w_fc1), b_fc1));//Fully connected

	Tensor h_fc1_drop = mLearn.Dropout(h_fc1, 0.8);                                   //Dropout

	Tensor w_fc2 = mLearn.Variable(0, 0.1, { 10,1024,1,1 });
	Tensor b_fc2 = mLearn.Variable(0.1, { 10,1,1,1 });

	Tensor y_fc2 = mLearn.add(mLearn.matmul(h_fc1_drop, w_fc2), b_fc2);
	Tensor y_conv = mLearn.Softmax(y_fc2);                                             //Calculate prediction result
	Tensor cross = mLearn.neg(mLearn.reduce_sum(mLearn.hadamard(y_, mLearn.log(y_conv)), 0));//Calculate loss

	graph.train.GradientDescent(0.01f, cross);                                         //Gradient descent for optimization

	free(b);
	free(c);
  ```
  ####test<br>
  ```
  double x_test[784 * 2] = { 0 };
	double y_test[10 * 2] = { 0 };
	mnist_test m_test;
	Tensor pre_bool = Tensor(0, { 1,1,1,2 });
	for (size_t i = 0; i < 40; i += 2)                                                 //Test 40 samples,2 samples as a goup
	{
		memset(y_test, 0, sizeof(y_test));
		m_test.getbatch(y_test, x_test, 2);
		Tensor &x = mLearn.refreshTensor(x_image);                                       //Reload new data for x_image
		x.replaceBatch(x_test, 784 * 2, 2);
		Tensor &y = mLearn.refreshTensor(y_);
		y.replaceBatch(y_test, 10 * 2, 2);
		mLearn.run(y_conv);                                                              //Calculate y_conv
		Tensor y_pre = mLearn.refreshTensor(y_conv);
		y_ = mLearn.refreshTensor(y_);                                                   //Refresh y_ value
		pre_bool.variableSplice(equal(argmax(y_pre, 0), argmax(y_, 0)));                 //Calculate the number of equals and recorded to pre_bool
	}
	pre_bool.variableMerge(0, 2);
	pre_bool.variableMerge(1, 2);
	
	
	cout << accuracy(pre_bool);                                                        //Output correct rate
  ```
  Current frame performance is poor,calculation is too slow and code bloated.
  I am also learning.Welcome to communicate with each other.The code will continue to improve.
