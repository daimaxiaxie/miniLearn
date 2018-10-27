#pragma once

#include "Tensor.h"


template<typename L>
void traverse(Shape shape, L lambda)                                        //Part adopts iterators traverse through will be better
{
	int w, h, c, v;
	for (v = 0; v < shape.variable; v++)
	{
		for (c = 0; c < shape.channels; c++)
		{
			for (h = 0; h < shape.height; h++)
			{
				for (w = 0; w < shape.width; w++)
				{
					lambda(w, h, c, v);
				}
			}
		}
	}
}

Tensor argmax(Tensor x, int axis)
{
	Tensor result = Tensor({ 1,1,1,1 });
	if (axis == 0)
	{
		result.reShape({ 1,x.getShape().height,x.getShape().channels,x.getShape().variable });
		for (size_t v = 0; v < x.getShape().variable; v++)
		{
			for (size_t c = 0; c < x.getShape().channels; c++)
			{
				for (size_t h = 0; h < x.getShape().height; h++)
				{
					int index = 0;
					double t = 0.0;
					for (size_t w = 0; w < x.getShape().width; w++)
					{
						if (x.get(w, h, c, v) > t)
						{
							t = x.get(w, h, c, v);
							index = w;
						}
					}
					result.add(index);
				}
			}
		}
	}
	else if (axis == 1)
	{
		result.reShape({ x.getShape().width,1,x.getShape().channels,x.getShape().variable });
		for (size_t v = 0; v < x.getShape().variable; v++)
		{
			for (size_t c = 0; c < x.getShape().channels; c++)
			{
				for (size_t w = 0; w < x.getShape().width; w++)
				{
					int index = 0;
					double t = 0.0;
					for (size_t h = 0; h < x.getShape().height; h++)
					{
						if (x.get(w, h, c, v) > t)
						{
							t = x.get(w, h, c, v);
							index = w;
						}
					}
					result.add(index);
				}
			}
		}
	}
	return result;
}

Tensor equal(Tensor x, Tensor y)
{
	if (x.getShape() != y.getShape())
	{
		return Tensor({ 0,0,0,0 });
	}
	Tensor result = Tensor({ 1,1,1,x.getShape().variable });
	int len = x.getShape().width*x.getShape().height*x.getShape().channels;
	double t = 1;
	for (size_t v = 0; v < x.getShape().variable; v++)
	{
		t = 1;
		for (size_t i = 0; i < len; i++)
		{
			if (x.get(i + len * v) != y.get(i + len * v))
			{
				t = 0;
				break;
			}
		}
		result.add(t);
	}
	return result;
}

float accuracy(Tensor x)
{
	int num = x.getShape().width*x.getShape().height*x.getShape().channels*x.getShape().variable;
	int correct = 0;
	for (size_t i = 0; i < num; i++)
	{
		if (x.get(i) == 1)
		{
			correct++;
		}
	}
	return (float)correct / num;
}




Tensor add(Tensor t1, Tensor t2)
{
	if (t1.getShape().width != t2.getShape().width || t1.getShape().height != t2.getShape().height || t1.getShape().channels != t2.getShape().channels)
	{
		if (t1.getShape().width == 1 && t1.getShape().height == 1)
		{
			int len = t2.getShape().width*t2.getShape().height;
			for (size_t v = 0; v < t1.getShape().variable; v++)
			{
				for (size_t c = 0; c < t1.getShape().channels; c++)
				{
					for (size_t i = 1; i < len; i++)
					{
						t1.insert(i + c * len + v * c * len, t1.get(c * len + v * c * len));
					}
				}
			}
			t1.reShape({ t2.getShape().width,t2.getShape().height,t1.getShape().channels,t1.getShape().variable });
		}
		else if (t2.getShape().width == 1 && t2.getShape().height == 1)
		{
			int len = t1.getShape().width*t1.getShape().height;
			for (size_t v = 0; v < t2.getShape().variable; v++)
			{
				for (size_t c = 0; c < t2.getShape().channels; c++)
				{
					for (size_t i = 1; i < len; i++)
					{
						t2.insert(i + c * len + v * c * len, t2.get(c * len + v * c * len));
					}
				}
			}
			t2.reShape({ t1.getShape().width,t1.getShape().height,t2.getShape().channels,t2.getShape().variable });
		}
		else
		{
			return Tensor({ 0,0,0,0 });
		}

	};
	if (t1.getShape().variable != t2.getShape().variable)
	{
		if (t1.getShape().variable == 1)
		{
			for (size_t i = 1; i < t2.getShape().variable; i++)
			{
				t1.variableCopy(0);
			}
		}
		else if (t2.getShape().variable == 1)
		{
			for (size_t i = 1; i < t1.getShape().variable; i++)
			{
				t2.variableCopy(0);
			}
		}
	}
	Tensor result = Tensor({ t1.getShape().width,t1.getShape().height,t1.getShape().channels,t1.getShape().variable });

	int i, j, l, c;
	for (l = 0; l < t1.getShape().variable; l++)
	{
		for (c = 0; c < t1.getShape().channels; c++)
		{
			for (i = 0; i < t1.getShape().height; i++)
			{
				for (j = 0; j < t1.getShape().width; j++)
				{
					result.add(t1.get(j, i, c, l) + t2.get(j, i, c, l));
				}
			}
		}
	}
	return result;
}

Tensor neg(Tensor t)     //opposite number
{
	Tensor result = Tensor({ t.getShape().width,t.getShape().height,t.getShape().channels,t.getShape().variable });
	int i, j, l, c;
	for (l = 0; l < t.getShape().variable; l++)
	{
		for (c = 0; c < t.getShape().channels; c++)
		{
			for (i = 0; i < t.getShape().height; i++)
			{
				for (j = 0; j < t.getShape().width; j++)
				{
					result.add(0 - t.get(j, i, c, l));
				}
			}
		}
	}
	return result;
}

Tensor abs(Tensor t)    //absolute value
{
	auto f = [&](int w, int h, int c, int v) {
		t.set(w, h, c, v, abs(t.get(w, h, c, v)));
	};
	traverse(t.getShape(), f);
	return t;
}

Tensor sub(Tensor t1, Tensor t2)
{
	return add(t1, neg(t2));
}

Tensor matmul(Tensor t1, Tensor t2)
{
	if (t1.getShape().width != t2.getShape().height)
	{
		return Tensor({ 0,0,0,0 });
	};
	if (t1.getShape().variable != t2.getShape().variable)
	{
		if (t1.getShape().variable == 1)
		{
			for (size_t i = 1; i < t2.getShape().variable; i++)
			{
				t1.variableCopy(0);
			}
		}
		else if (t2.getShape().variable == 1)
		{
			for (size_t i = 1; i < t1.getShape().variable; i++)
			{
				t2.variableCopy(0);
			}
		}
	}
	Tensor result = Tensor({ t2.getShape().width,t1.getShape().height,t2.getShape().channels,t2.getShape().variable });

	int i, j, k, l, c;
	//t1.fill_clear(11199.5468750000000);
	for ( l = 0; l < t2.getShape().variable; l++)
	{
		for (c = 0; c < t2.getShape().channels; c++)
		{
			for (i = 0; i < t1.getShape().height; i++)
			{
				for (j = 0; j < t2.getShape().width; j++)
				{
					double x = 0.0;
					for (k = 0; k < t1.getShape().width; k++)
					{
						x = t1.get(k, i, c, l)*t2.get(j, k, c, l) + x;
					}
					result.add(x);
				}

			}
		}
	}

	return result;
}

Tensor matmul(Tensor t, float a)
{
	Shape shape = t.getShape();
	int i, j, k, l;
	for (i = 0; i < shape.variable; i++)
	{
		for (j = 0; j < shape.channels; j++)
		{
			for (k = 0; k < shape.height; k++)
			{
				for (l = 0; l < shape.width; l++)
				{
					t.set(l, k, j, i, t.get(l, k, j, i)*a);
				}
			}
		}
	}
	return t;
}

Tensor hadamard(Tensor t1, Tensor t2)
{
	if (t1.getShape().width != t2.getShape().width&&t1.getShape().height != t2.getShape().height)
	{
		return Tensor({ 0,0,0,0 });
	};
	Tensor result = Tensor({ t1.getShape().width,t1.getShape().height,t1.getShape().channels,t1.getShape().variable });

	int i, j, l, c;
	for (l = 0; l < t1.getShape().variable; l++)
	{
		for (c = 0; c < t1.getShape().channels; c++)
		{
			for (i = 0; i < t1.getShape().height; i++)
			{
				for (j = 0; j < t1.getShape().width; j++)
				{
					result.add(t1.get(j, i, c, l)*t2.get(j, i, c, l));
				}
			}
		}
	}
	return result;
}

Tensor exp(Tensor input)
{
	Shape shape = input.getShape();
	int i, j, k, l;
	for (i = 0; i < shape.variable; i++)
	{
		for (j = 0; j < shape.channels; j++)
		{
			for (k = 0; k < shape.height; k++)
			{
				for (l = 0; l < shape.width; l++)
				{
					input.set(l, k, j, i, exp(input.get(l, k, j, i)));
				}
			}
		}
	}
	return input;
}

Tensor log(Tensor input)
{
	int i, j, k, l;
	for (i = 0; i < input.getShape().variable; i++)
	{
		for (j = 0; j < input.getShape().channels; j++)
		{
			for (k = 0; k < input.getShape().height; k++)
			{
				for (l = 0; l < input.getShape().width; l++)
				{
					double value = input.get(l, k, j, i);
					double ln = 0.0;
					ln = log(value);
					input.set(l, k, j, i, ln);
				}
			}
		}
	}
	return input;
}

Tensor reduce_sum(Tensor input, int axis)
{
	if (axis < 0 || axis > 3)
	{
		return Tensor({ 0,0,0,0 });
	}
	Shape shape = input.getShape();
	int i, j, k, l;
	Tensor result = Tensor({ 1,1,1,1 });
	if (axis == 0)
	{
		result.reShape({ 1,shape.height,shape.channels,shape.variable });
		for (i = 0; i < shape.variable; i++)
		{
			for (j = 0; j < shape.channels; j++)
			{
				for (k = 0; k < shape.height; k++)
				{
					double temp = 0.0;
					for (l = 0; l < shape.width; l++)
					{
						temp += input.get(l, k, j, i);
					}
					result.add(temp);
				}
			}
		}
	}
	else if (axis == 1)
	{
		result.reShape({ shape.width,1,shape.channels,shape.variable });
		for (i = 0; i < shape.variable; i++)
		{
			for (j = 0; j < shape.channels; j++)
			{
				for (l = 0; l < shape.width; l++)
				{
					double temp = 0.0;
					for (k = 0; k < shape.height; k++)
					{
						temp += input.get(l, k, j, i);
					}
					result.add(temp);
				}
			}
		}
	}
	else if (axis == 2)
	{
		result.reShape({ shape.width,shape.height,1,shape.variable });
		for (i = 0; i < shape.variable; i++)
		{
			for (k = 0; k < shape.height; k++)
			{
				for (l = 0; l < shape.width; l++)
				{
					double temp = 0.0;
					for (j = 0; j < shape.channels; j++)
					{
						temp += input.get(l, k, j, i);
					}
					result.add(temp);
				}
			}
		}
	}
	else if (axis == 3)
	{
		result.reShape({ shape.width,shape.height,shape.channels,1 });
		for (j = 0; j < shape.channels; j++)
		{
			for (k = 0; k < shape.height; k++)
			{
				for (l = 0; l < shape.width; l++)
				{
					double temp = 0.0;
					for (i = 0; i < shape.variable; i++)
					{
						temp += input.get(l, k, j, i);
					}
					result.add(temp);
				}
			}
		}
	}
	return result;
}

Tensor reshape(Tensor input, Shape shape)
{
	input.reShape(shape);
	return input;
}

Tensor conv2d(Tensor input, Tensor filter, Shape strides, padding p)
{
	if (!(input.isInit() && filter.isInit() && strides.checkShape()))
	{
		return Tensor({ 0,0,0,0 });
	}
	Shape input_shape = input.getShape();
	Shape filter_shape = filter.getShape();
	if (input_shape.width < filter_shape.width || input_shape.height < filter_shape.height || input_shape.channels != filter_shape.channels)
	{
		return Tensor({ 0,0,0,0 });
	}

	int inWidth = input_shape.width;
	int inHeight = input_shape.height;

	/*
	int i = (input_shape.width - filter_shape.width) % strides.width;
	int j = (input_shape.height - filter_shape.height) % strides.height;
	if (i != 0 || j != 0)
	{
		for (size_t t = 0; t < (unsigned)strides.width - i; t++)
		{
			input.padding2d(input_shape.width, -1);
			input_shape = input.getShape();
		}
		for (size_t t = 0; t < (unsigned)strides.height - j; t++)
		{
			input.padding2d(-1, input_shape.height);
			input_shape = input.getShape();
		}
	}*/

	bool paddingH, paddingV;                                                            //control padding position :: H0:left,H1:right,V0:up,V1:down
	paddingH = true;
	paddingV = true;
	int width = 0;//input_shape.width / strides.width;
	int height = 0;//input_shape.height / strides.height;
	if (p == padding::SAME)
	{
		if (width < input_shape.width)
		{

		
			size_t pad_along_width = 0;
			if (inWidth % strides.width == 0)
			{
				pad_along_width = max(filter_shape.width - strides.width, 0);
			}
			else
			{
				pad_along_width = max(filter_shape.width - (inWidth % strides.width), 0);
			}
			
			for (size_t t = 0; t < pad_along_width; t++)
			{
				if (paddingH)
				{
					input.padding2d(input_shape.width, -1);
				}
				else
				{
					input.padding2d(0, -1);
				}
				paddingH = !paddingH;
				input_shape = input.getShape();
			}
		}
		if (height < inHeight)
		{

		
			size_t pad_along_height = 0;
			if (inHeight % strides.height == 0)
			{
				pad_along_height = max(filter_shape.height - strides.height, 0);
			}
			else
			{
				pad_along_height = max(filter_shape.height - (inHeight % strides.height), 0);
			}
			
			for (size_t t = 0; t < pad_along_height; t++)
			{
				if (paddingV)
				{
					input.padding2d(-1, input_shape.height);
				}
				else
				{
					input.padding2d(-1, 0);
				}
				paddingV = !paddingV;
				input_shape = input.getShape();
			}
		}
		width = static_cast<int> (ceil(inWidth / (double)strides.width));
		height = static_cast<int> (ceil(inHeight / (double)strides.height));
	}
	else
	{
		width = ceil((input_shape.width - filter_shape.width + 1) / (double)strides.width);
		height = ceil((input_shape.height - filter_shape.height + 1) / (double)strides.height);

	}
	Tensor result = Tensor({ width,height,filter_shape.variable * input_shape.channels,input_shape.variable });
	int outChannels = 0;

	int i, j;
	for (i = 0; i < input_shape.variable; i++)
	{
		for (outChannels = 0; outChannels < filter_shape.variable; outChannels++)
		{
			for (j = 0; j < input_shape.channels; j++)
			{
				int t1, t2, p;
				double temp_d;
				//long long temp_l;
				for (t1 = 0; t1 <= input_shape.height - filter_shape.height; t1 += strides.height)      //(height - 1) * strides.height + filter_shape.height - strides.height
				{
					for (t2 = 0; t2 <= input_shape.width - filter_shape.width; t2 += strides.width)
					{
						temp_d = 0;
						for (p = 0; p < filter_shape.width * filter_shape.height; p++)
						{
							double x, y;
							int a, b;
							y = filter.get(p + j * filter_shape.width * filter_shape.height + outChannels * filter_shape.channels * filter_shape.height * filter_shape.width);
							a = p / filter_shape.width;
							b = p % filter_shape.width;
							x = input.get(t2 + b, t1 + a, j, i);
							temp_d += x * y;
							//if(t2==0&&t1>4)std::cout << x1 << "X" << y1 << "\n";
						}
						result.add(temp_d);
					}
				}
			}
		}
	}
	//std::cout << result;

	for (outChannels = 0; outChannels < filter_shape.variable; outChannels++)
	{
		for (j = 1; j < input_shape.channels; j++)
		{
			result.channelsMerge(outChannels, outChannels + 1);
		}
	}
	return result;
}

Tensor max_pool(Tensor input, Shape ksize, Shape strides,padding pad)
{
	Shape input_shape = input.getShape();
	int inWidth = input.getShape().width;
	int inHeight = input.getShape().height;

	bool paddingH, paddingV;                                                            //control padding position :: H0:left,H1:right,V0:up,V1:down
	paddingH = true;
	paddingV = true;
	int width = 0;//input_shape.width / strides.width;
	int height = 0;//input_shape.height / strides.height;
	if (pad == padding::SAME)
	{
		if (width < ksize.width)
		{


			size_t pad_along_width = 0;
			if (inWidth % strides.width == 0)
			{
				pad_along_width = max(ksize.width - strides.width, 0);
			}
			else
			{
				pad_along_width = max(ksize.width - (inWidth % strides.width), 0);
			}

			for (size_t t = 0; t < pad_along_width; t++)
			{
				if (paddingH)
				{
					input.padding2d(input_shape.width, -1);
				}
				else
				{
					input.padding2d(0, -1);
				}
				paddingH = !paddingH;
				input_shape = input.getShape();
			}
		}
		if (height < inHeight)
		{


			size_t pad_along_height = 0;
			if (inHeight % strides.height == 0)
			{
				pad_along_height = max(ksize.height - strides.height, 0);
			}
			else
			{
				pad_along_height = max(ksize.height - (inHeight % strides.height), 0);
			}

			for (size_t t = 0; t < pad_along_height; t++)
			{
				if (paddingV)
				{
					input.padding2d(-1, input_shape.height);
				}
				else
				{
					input.padding2d(-1, 0);
				}
				paddingV = !paddingV;
				input_shape = input.getShape();
			}
		}
		width = static_cast<int> (ceil(inWidth / (double)strides.width));
		height = static_cast<int> (ceil(inHeight / (double)strides.height));
	}
	else
	{
		width = ceil((input_shape.width - ksize.width + 1) / (double)strides.width);
		height = ceil((input_shape.height - ksize.height + 1) / (double)strides.height);
	}

	Tensor pool = Tensor({ width, height, input_shape.channels, input_shape.variable });

	int i, j;
	for (i = 0; i < input_shape.variable; i++)
	{
		for (j = 0; j < input_shape.channels; j++)
		{
			int t1, t2, p;
			double temp = 0;
			for (t1 = 0; t1 <= input_shape.height - ksize.height; t1 += strides.height)
			{
				for (t2 = 0; t2 <= input_shape.width - ksize.width; t2 += strides.width)
				{
					temp = 0;
					double *arr = new double[ksize.height*ksize.width];
					for (p = 0; p < ksize.height * ksize.width; p++)
					{
						int a, b;
						double x = 0;
						a = p / ksize.width;
						b = p % ksize.width;
						x = input.get(t2 + b, t1 + a, j, i);
						arr[p] = x;
					}
					temp = max(ksize.width*ksize.height, arr);
					pool.add(temp);
				}
			}
		}
	}

	return pool;
}

Tensor average_pool(Tensor input,Shape ksize,Shape strides,padding pad)
{
	return input;
}

Tensor relu(Tensor input)
{
	Shape input_shape = input.getShape();

	Tensor result = Tensor(input_shape);

	int i, j, k, l;
	for (i = 0; i < input_shape.variable; i++)
	{
		for (j = 0; j < input_shape.channels; j++)
		{
			for (k = 0; k < input_shape.height; k++)
			{
				for (l = 0; l < input_shape.width; l++)
				{
					double temp = input.get(l, k, j, i);
					result.add(max(temp, 0.0));
				}
			}
		}
	}
	return result;
}

Tensor dropout(Tensor input, float keep_prob)
{
	if (keep_prob < 0 || keep_prob > 1)
	{
		return Tensor({ 0,0,0,0 });
	}

	Shape shape = input.getShape();

	float prob = keep_prob * 50;
	int i, j, k, l;
	srand((unsigned)time(NULL));
	for (i = 0; i < shape.variable; i++)
	{
		for (j = 0; j < shape.channels; j++)
		{
			for (k = 0; k < shape.height; k++)
			{
				for (l = 0; l < shape.width; l++)
				{
					int random = rand() % 50;
					if (prob > random)
					{
						double t = input.get(l, k, j, i) / (double)keep_prob;
						input.set(l, k, j, i, t);
					}
					else
					{
						input.set(l, k, j, i, 0);
					}
				}
			}
		}
	}
	return input;
}

Tensor softmax(Tensor input)
{
	Shape shape = input.getShape();
	int i, j, k, l;
	
	for (i = 0; i < shape.variable; i++)
	{
		for (j = 0; j < shape.channels; j++)
		{
			for (k = 0; k < shape.height; k++)
			{
				double d;
				double *t = new double[shape.width];
				for (l = 0; l < shape.width; l++)
				{
					t[l] = input.get(l, k, 0, i);
				}
				d = max(shape.width, t);
				for (l = 0; l < shape.width; l++)
				{
					input.set(l, k, 0, i, input.get(l, k, 0, i) - d);
				}
			}
		}
	}
	if (shape.height == 1 && shape.channels == 1)
	{
		Tensor temp = reduce_sum(exp(input), 0);
		for (i = 0; i < shape.variable; i++)
		{
			double sum = temp.get(0, 0, 0, i);
			for (l = 0; l < shape.width; l++)
			{
				if (isnan(sum))
				{
					input.set(1, 0, 0, i, 0);
					continue;
				}
				double e = exp(input.get(l, 0, 0, i));
				input.set(l, 0, 0, i, e / sum);
			}
		}
	}
	else if (shape.channels ==1)
	{
		Tensor temp = reduce_sum(exp(input), 0);
		for (i = 0; i < shape.variable; i++)
		{
			for (j = 0; j < shape.channels; j++)
			{
				for (k = 0; k < shape.height; k++)
				{
					double sum = temp.get(0, k, 0, i);
					for (l = 0; l < shape.width; l++)
					{
						if (isnan(sum))
						{
							input.set(1, 0, 0, i, 0);
							continue;
						}
						double e = exp(input.get(l, k, 0, i));
						input.set(l, k, 0, i, e / sum);
					}
				}
			}
		}
	}
	else
	{
		Tensor temp = reduce_sum(exp(input), 2);
		for (i = 0; i < shape.variable; i++)
		{
			for (j = 0; j < shape.channels; j++)
			{
				for (k = 0; k < shape.height; k++)
				{
					for (l = 0; l < shape.width; l++)
					{
						double sum = temp.get(l, k, 0, i);
						if (isnan(sum))
						{
							input.set(1, 0, 0, i, 0);
							continue;
						}
						double e = exp(input.get(l, k, j, i));
						input.set(l, k, j, i, e / sum);
					}
				}
			}
		}
		//Maybe is error
	}
	return input;
}

Tensor inverse_maxpool(Tensor delta, Tensor pool, Tensor input)
{
	Tensor result = Tensor(input.getShape());
	Shape input_shape = input.getShape();
	Shape ksize = pool.getPar()->ksize;
	Shape strides = pool.getPar()->strides;
	padding pad = pool.getPar()->pad;
	int w, h, c, v, p, index;
	index = 0;
	int *pos = new int[pool.getShape().width*pool.getShape().height*pool.getShape().channels*pool.getShape().variable];
	int inWidth = input.getShape().width;
	int inHeight = input.getShape().height;

	bool paddingH, paddingV;
	paddingH = true;
	paddingV = true;
	int width = 0;
	int height = 0;
	if (pad == padding::SAME)
	{
		if (width < ksize.width)
		{


			size_t pad_along_width = 0;
			if (inWidth % strides.width == 0)
			{
				pad_along_width = max(ksize.width - strides.width, 0);
			}
			else
			{
				pad_along_width = max(ksize.width - (inWidth % strides.width), 0);
			}

			for (size_t t = 0; t < pad_along_width; t++)
			{
				if (paddingH)
				{
					input.padding2d(input_shape.width, -1);
				}
				else
				{
					input.padding2d(0, -1);
				}
				paddingH = !paddingH;
				input_shape = input.getShape();
			}
		}
		if (height < inHeight)
		{


			size_t pad_along_height = 0;
			if (inHeight % strides.height == 0)
			{
				pad_along_height = max(ksize.height - strides.height, 0);
			}
			else
			{
				pad_along_height = max(ksize.height - (inHeight % strides.height), 0);
			}

			for (size_t t = 0; t < pad_along_height; t++)
			{
				if (paddingV)
				{
					input.padding2d(-1, input_shape.height);
				}
				else
				{
					input.padding2d(-1, 0);
				}
				paddingV = !paddingV;
				input_shape = input.getShape();
			}
		}
		width = static_cast<int> (ceil(inWidth / (double)strides.width));
		height = static_cast<int> (ceil(inHeight / (double)strides.height));
	}
	else
	{
		width = ceil((input_shape.width - ksize.width + 1) / (double)strides.width);
		height = ceil((input_shape.height - ksize.height + 1) / (double)strides.height);
	}

	for (v = 0; v < input.getShape().variable; v++)
	{
		for (c = 0; c < input.getShape().channels; c++)
		{
			for (h = 0; h <= input.getShape().height - ksize.height; h += strides.height)
			{
				for (w = 0; w <= input.getShape().width - ksize.width; w += strides.width)
				{
					double temp = 0;
					int i = 0;
					for (p = 0; p < ksize.height * ksize.width; p++)
					{
						int a, b;
						double x = 0;
						a = p / ksize.width;
						b = p % ksize.width;
						x = input.get(w + b, h + a, c, v);
						if (temp < x)
						{
							temp = x;
							i = p;
						}
					}
					pos[index] = i;
					index++;
				}
			}
		}
	}

	index = 0;
	result.fill_clear(0);
	for (v = 0; v < result.getShape().variable; v++)
	{
		for (c = 0; c < result.getShape().channels; c++)
		{
			for (h = 0; h <= input.getShape().height - ksize.height; h += strides.height)
			{
				for (w = 0; w <= input.getShape().width - ksize.width; w += strides.width)
				{
					int a, b;
					p = pos[index];
					a = p / ksize.width;
					b = p % ksize.width;
					result.set(w + b, h + a, c, v, delta.get(index));
					index++;
				}
			}
		}
	}
	return result;
}

Tensor inverse_avgpool(Tensor delta, Tensor pool, Tensor input)
{
	Tensor result = Tensor(input.getShape());
	return result;
}

Tensor inverse_conv_weight(Tensor w, Tensor input, Tensor delta, Shape strides, padding pad)
{
	int weight, height;
	weight = strides.width*(w.getShape().width - 1) + delta.getShape().width;
	height = strides.height*(w.getShape().height - 1) + delta.getShape().height;
	bool pad_w, pad_h;
	pad_w = true;
	pad_h = true;
	for (size_t i = 0; i < input.getShape().width - weight; i++)
	{
		if (pad_w)
		{
			input.padding2d(input.getShape().width, -1);
		}
		else
		{
			input.padding2d(0, -1);
		}
		pad_w = !pad_w;
	}
	for (size_t i = 0; i < input.getShape().height - height; i++)
	{
		if (pad_h)
		{
			input.padding2d(-1, input.getShape().height);
		}
		else
		{
			input.padding2d(-1, 0);
		}
		pad_h = !pad_h;
	}

	int ch = delta.getShape().channels;
	for (size_t c_delta = 0; c_delta < ch; c_delta++)
	{
		for (size_t c = 1; c < w.getShape().channels; c++)
		{
			delta.channelsCopy(c_delta*w.getShape().channels);
		}
	}
	
	delta.reShape({ delta.getShape().width,delta.getShape().height,w.getShape().channels,w.getShape().variable });

	int outChannels = 0;
	Tensor result = Tensor(w.getShape());
	int j;
	//for (i = 0; i < input.getShape().variable; i++)
	//{
		for (outChannels = 0; outChannels < delta.getShape().variable; outChannels++)
		{
			for (j = 0; j < input.getShape().channels; j++)
			{
				int t1, t2, p;
				double temp_d;
				for (t1 = 0; t1 <= input.getShape().height - delta.getShape().height; t1 += strides.height)      //(height - 1) * strides.height + filter_shape.height - strides.height
				{
					for (t2 = 0; t2 <= input.getShape().width - delta.getShape().width; t2 += strides.width)
					{
						temp_d = 0;
						for (p = 0; p < delta.getShape().width * delta.getShape().height; p++)
						{
							double x, y;
							int a, b;
							y = delta.get(p + j * delta.getShape().width * delta.getShape().height + outChannels * delta.getShape().channels * delta.getShape().height * delta.getShape().width);
							a = p / delta.getShape().width;
							b = p % delta.getShape().width;
							x = input.get(t2 + b, t1 + a, j, 0);
							temp_d += x * y;
						}
						result.add(temp_d);
					}
				}
			}
		}
	//}
	return result;
}

Tensor inverse_conv(Tensor input, Tensor delta, Tensor weight, Shape strides)
{
	int weight_p, height_p;
	weight_p = strides.width*(input.getShape().width - 1) + weight.getShape().width;
	height_p = strides.height*(input.getShape().height - 1) + weight.getShape().height;
	bool pad_w, pad_h;
	pad_w = false;
	pad_h = false;
	for (size_t i = 0; i < delta.getShape().width - weight_p; i++)
	{
		if (pad_w)
		{
			delta.padding2d(delta.getShape().width, -1);
		}
		else
		{
			delta.padding2d(0, -1);
		}
		pad_w = !pad_w;
	}
	for (size_t i = 0; i < delta.getShape().height - height_p; i++)
	{
		if (pad_h)
		{
			delta.padding2d(-1, delta.getShape().height);
		}
		else
		{
			delta.padding2d(-1, 0);
		}
		pad_h = !pad_h;
	}

	int ch=delta.getShape().channels;
	for (size_t c_delta = 0; c_delta < ch; c_delta++)
	{
		for (size_t c = 1; c < weight.getShape().channels; c++)
		{
			delta.channelsCopy(c_delta*weight.getShape().channels);
		}
	}
	delta.reShape({ delta.getShape().width,delta.getShape().height,weight.getShape().channels,weight.getShape().variable });

	weight.rota2D();
	Tensor result = Tensor({ input.getShape().width,input.getShape().height,delta.getShape().channels,delta.getShape().variable });
	int i, Channels;
	for (i = 0; i < delta.getShape().variable; i++)
    {
		for (Channels = 0; Channels < delta.getShape().channels; Channels++)
		{
			int t1, t2, p;
			double temp_d;
			for (t1 = 0; t1 <= delta.getShape().height - weight.getShape().height; t1 += strides.height)
			{
				for (t2 = 0; t2 <= delta.getShape().width - weight.getShape().width; t2 += strides.width)
				{
					temp_d = 0;
					for (p = 0; p < weight.getShape().width * weight.getShape().height; p++)
					{
						double x, y;
						int a, b;
						y = weight.get(p +  Channels* weight.getShape().width * weight.getShape().height + i * weight.getShape().channels * weight.getShape().height * weight.getShape().width);
						a = p / weight.getShape().width;
						b = p % weight.getShape().width;
						x = delta.get(t2 + b, t1 + a, Channels, i);
						temp_d += x * y;
					}
					result.add(temp_d);
				}
			}
		}
	}

	for (Channels = 1; Channels < result.getShape().variable; Channels++)
	{
		result.variableMerge(0, Channels);
	}

	return result;
}

Tensor inverse_dropout(Tensor input, Tensor output, Tensor delta)
{
	auto f = [&](int w, int h, int c, int v) {
		if (input.get(w, h, c, v) != 0 && output.get(w, h, c, v) == 0)
		{
			delta.set(w, h, c, v, 0.0);
		}
	};
	traverse(output.getShape(), f);
	return delta;
}
