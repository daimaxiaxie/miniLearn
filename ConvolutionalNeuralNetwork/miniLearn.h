#pragma once

#include "struct/Graph.h"

class miniLearn
{
public:
	Graph& CreateGraph()
	{
		return graph;
	}


	Tensor& Variable(Shape shape)
	{
		Tensor t = Tensor(shape);
		t.setKind(VertexKind::Variable);
		return graph.addVertex(t);
	}

	Tensor& Variable(double data[], Shape shape)
	{
		Tensor t = Tensor(data, shape.width*shape.height*shape.channels*shape.variable, shape);
		t.setKind(VertexKind::Variable);
		return graph.addVertex(t);
	}

	Tensor& Variable(float value, Shape shape)
	{
		Tensor t = Tensor(value, shape);
		t.setKind(VertexKind::Variable);
		return graph.addVertex(t);
	}

	Tensor& Variable(float mean, float stddev, Shape shape)
	{
		Tensor t = Tensor(mean, stddev, shape);
		t.setKind(VertexKind::Variable);
		return graph.addVertex(t);
	}

	Tensor& Constant(double data[], int size, Shape shape)
	{
		Tensor t = Tensor(data, size, shape);
		t.setKind(VertexKind::Constant);
		t.setbatch(shape.variable);
		return graph.addVertex(t);
	}

	Tensor& Constant(double data[], int size, Shape shape, int batchs)
	{
		//shape.variable = 1;
		Tensor t = Tensor(data, size, shape);
		t.setKind(VertexKind::Constant);
		t.setbatch(batchs);
		return graph.addVertex(t);
	}

	Tensor& refreshTensor(Tensor x)
	{
		return graph.getVertex(x.getNodeID());
	}

	Tensor& Convolution(Tensor &input, Tensor &filter, Shape strides, padding pad)
	{
		int width = 0, height = 0;
		if (pad == padding::SAME)
		{
			width = static_cast<int> (ceil(input.getShape().width / (double)strides.width));
			height = static_cast<int> (ceil(input.getShape().height / (double)strides.height));
		}
		else
		{
			width = ceil((input.getShape().width - filter.getShape().width + 1) / (double)strides.width);
			height = ceil((input.getShape().height - filter.getShape().height + 1) / (double)strides.height);
		}
		Tensor t = Tensor({ width,height,filter.getShape().variable,input.getShape().variable });
		t.setKind(VertexKind::Convolution);
		return graph.addVertex(t, input, filter, strides, pad);
	}

	Tensor& MaxPool(Tensor &input, Shape ksize, Shape strides, padding pad)
	{
		int width = 0, height = 0;
		if (pad == padding::SAME)
		{
			width = static_cast<int> (ceil(input.getShape().width / (double)strides.width));
			height = static_cast<int> (ceil(input.getShape().height / (double)strides.height));
		}
		else
		{
			width = ceil((input.getShape().width - ksize.width + 1) / (double)strides.width);
			height = ceil((input.getShape().height - ksize.height + 1) / (double)strides.height);
		}
		Tensor t = Tensor({ width,height,input.getShape().channels,input.getShape().variable });
		t.setKind(VertexKind::MaxPooling);
		return graph.addVertex(t, input, ksize, strides, pad);
	}

	/*
	Tensor& AvgPool(Tensor &input, Shape ksize, Shape strides, padding pad)
	{
		int width = 0, height = 0;
		if (pad == padding::SAME)
		{
			width = static_cast<int> (ceil(input.getShape().width / (double)strides.width));
			height = static_cast<int> (ceil(input.getShape().height / (double)strides.height));
		}
		else
		{
			width = ceil((input.getShape().width - ksize.width + 1) / (double)strides.width);
			height = ceil((input.getShape().height - ksize.height + 1) / (double)strides.height);
		}
		Tensor t = Tensor({ width,height,input.getShape().channels,input.getShape().variable });
		t.setKind(VertexKind::MaxPooling);
		return graph.addVertex(t, input, ksize, strides, pad);
	}
    */

	Tensor& Relu(Tensor &input)
	{
		Tensor t = Tensor(input.getShape());
		t.setKind(VertexKind::Relu);
		return graph.addVertex(t,input);
	}

	Tensor& Dropout(Tensor &input, float keep_prob)
	{
		Tensor t = Tensor(input.getShape());
		t.setKind(VertexKind::Dropout);
		return graph.addVertex(t, input, keep_prob);
	}

	Tensor& Softmax(Tensor &input)
	{
		Tensor t = Tensor(input.getShape());
		t.setKind(VertexKind::Softmax);
		return graph.addVertex(t, input);
	}


	Tensor& reduce_sum(Tensor &input, int axis)
	{
		Tensor t = Tensor(input.getShape());
		t.setKind(VertexKind::Reduce_sum);
		return graph.addVertex(t, input,axis);
	}

	Tensor& reshape(Tensor &input, Shape newShape)
	{
		Tensor t = Tensor(newShape);
		t.setKind(VertexKind::ReShape);
		return graph.addVertex(t, input, newShape);
	}

	Tensor& add(Tensor t1, Tensor t2)
	{
		Tensor tensor = Tensor({ t2.getShape().width,t1.getShape().height,t1.getShape().channels,t1.getShape().variable });
		tensor.setKind(VertexKind::Addition);
		return graph.addVertex(tensor, t1, t2);
	}
	
	Tensor& matmul(Tensor t1, Tensor t2)
	{
		Tensor tensor = Tensor({ t2.getShape().width,t1.getShape().height,t1.getShape().channels,t1.getShape().variable });
		tensor.setKind(VertexKind::Multiplication);
		return graph.addVertex(tensor, t1, t2);
	}

	Tensor& hadamard(Tensor t1, Tensor t2)
	{
		Tensor tensor = Tensor({ max(t1.getShape().width,t2.getShape().width),max(t1.getShape().height,t2.getShape().height),t1.getShape().channels,t1.getShape().variable });
		tensor.setKind(VertexKind::Hadamard);
		return graph.addVertex(tensor, t1, t2);
	}

	Tensor& neg(Tensor input)
	{
		Tensor t = Tensor(input.getShape());
		t.setKind(VertexKind::Neg);
		return graph.addVertex(t, input);
	}

	Tensor& exp(Tensor input)
	{
		Tensor t = Tensor(input.getShape());
		t.setKind(VertexKind::Exp);
		return graph.addVertex(t, input);
	}

	Tensor& log(Tensor input)
	{
		Tensor t = Tensor(input.getShape());
		t.setKind(VertexKind::Logarithm);
		return graph.addVertex(t, input);
	}

	Tensor& run(Tensor input)
	{
		return graph.run(input);
	}

private:
	Graph graph;
};
