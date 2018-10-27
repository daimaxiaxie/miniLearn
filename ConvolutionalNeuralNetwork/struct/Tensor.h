#pragma once

#include "head.h"
#include "enum.h"
#include "baseOP.h"

struct Shape
{
	int width;
	int height;
	int channels;
	int variable;

	bool checkShape()
	{
		if (width&&height&&channels&&variable)
		{
			return true;
		}
		return false;
	}

	bool operator==(Shape shape)
	{
		if (width == shape.width&&height == shape.height&&channels == shape.channels&&variable == shape.variable)
		{
			return true;
		}
		return false;
	}

	bool operator!=(Shape shape)
	{
		return !operator==(shape);
	}
};

struct Parameter
{
	Shape strides;
	padding pad;

	Shape ksize;

	float keep_prob;

	int axis;

	Shape shape;
	Shape newshape;
};


class Tensor
{
public:
	Tensor(Shape shape)
	{
		if (!shape.checkShape())
		{
			throw std::bad_array_new_length();
			return;
		}
		this->shape = shape;
		length = shape.width*shape.height*shape.channels*shape.variable;
		this->batch = shape.variable;
		//length = 0;
		firstarc = nullptr;
		inversearc = nullptr;
		par = nullptr;
	}

	Tensor(float value, Shape shape)
	{
		if (!shape.checkShape())
		{
			throw std::bad_array_new_length();
			return;
		}
		int size = shape.width*shape.height*shape.channels*shape.variable;
		//double t = value * 1.0000000000000000;
		for (int i = 0; i < size; i++)
		{
			tensor.push_back(value);
		}
		length = size;
		this->shape = shape;
		this->batch = shape.variable;

		firstarc = nullptr;
		inversearc = nullptr;
		par = nullptr;
	}
	
	Tensor(float mean, float stddev, Shape shape)
	{
		if (!shape.checkShape())
		{
			throw std::bad_array_new_length();
			return;
		}
		int size = shape.width*shape.height*shape.channels*shape.variable;
		double rand = 0.0;
		for (int i = 0; i < size; i++)
		{
			rand = rand_back(mean, stddev);
			while (fabs(rand - mean) > (2 * stddev))
			{
				rand = rand_back(mean, stddev);
			}
			tensor.push_back(rand);
		}
		length = size;
		this->shape = shape;
		this->batch = shape.variable;

		firstarc = nullptr;
		inversearc = nullptr;
		par = nullptr;
	}

	Tensor(double data[], int size, Shape shape)
	{
		if (!shape.checkShape())
		{
			throw std::bad_array_new_length();
			return;
		}
		for (int i = 0; i < size; i++)
		{
			tensor.push_back(data[i]);
		}
		length = shape.width*shape.height*shape.channels*shape.variable;
		this->shape = shape;
		this->batch = shape.variable;

		firstarc = nullptr;
		inversearc = nullptr;
		par = nullptr;
	}

	template <typename T>
	Tensor(T& data, Shape shape)
	{
		if (!shape.checkShape())
		{
			throw std::bad_array_new_length();
			return;
		}
		length = sizeof(data) / sizeof(data[0]);
		if (strstr(typeid(data).name(), "double") != NULL)
		{
			for (int i = 0; i < length; i++)
			{
				tensor.push_back(data[i]);
			}
		}
		else if (strstr(typeid(data).name(), "int") != NULL)
		{
			for (int i = 0; i < length; i++)
			{
				tensor.push_back((double)data[i]);
			}
		}
		else if (strstr(typeid(data).name(), "float") != NULL)
		{
			for (int i = 0; i < length; i++)
			{
				tensor.push_back((double)data[i]);
			}
		}
		else
		{
			throw std::bad_typeid();
			return;
		}
		length = shape.width*shape.height*shape.channels*shape.variable;
		this->shape = shape;

		firstarc = nullptr;
		inversearc = nullptr;
		par = nullptr;
	}

	~Tensor()
	{
		tensor.clear();
		tensor.~vector();
	}

	bool isInit()
	{
		if (tensor.empty())
		{
			return false;
		}
		if (!shape.checkShape())
		{
			return false;
		}
		if (length < 2)
		{
			return false;
		}
		return true;
	}

    int getLength()
	{
		return length;
	}

	Shape getShape()
	{
		return shape;
	}

	int getSize()
	{
		return tensor.size();
	}

	double get(int i)
	{

		return tensor[i];
	}

	double get(int width, int height, int channel, int batch)
	{
		return tensor[width + height * shape.width + channel * shape.height * shape.width + batch * shape.channels * shape.height * shape.width];
	}

	int getbatch()
	{
		return batch;
	}

	void set(int i, double value)
	{
		tensor[i] = value;
	}

	void set(int width, int height, int channel, int batch, double value)
	{
		tensor[width + height * shape.width + channel * shape.height * shape.width + batch * shape.channels * shape.height * shape.width] = value;
	}

	void setbatch(int batchs = 1)
	{
		batch = batchs;
	}

	void remove(int width, int height, int channel, int batch)
	{
		int pos = width + height * shape.width + channel * shape.height * shape.width + batch * shape.channels * shape.height * shape.width;
		tensor.erase(tensor.begin() + pos);
	}

	void remove(int begin, int end)
	{
		tensor.erase(tensor.begin() + begin, tensor.begin() + end);
	}

	void reShape(Shape s)
	{
		if (!s.checkShape())
		{
			return;
		}
		shape = s;
		batch = shape.variable;
	}

	void add(double t)
	{
		tensor.push_back(t);
	}

	void fill_clear(double t)
	{
		int w, h, c, v;
		tensor.clear();
		for (v = 0; v < shape.variable; v++)
		{
			for (c = 0; c < shape.channels; c++)
			{
				for (h = 0; h < shape.height; h++)
				{
					for (w = 0; w < shape.width; w++)
					{
						add(t);
					}
				}
			}
		}
	}

	void fill(double t)
	{
		length = shape.width*shape.height*shape.channels*shape.variable;
		int len = length - tensor.size();
		for (size_t i= 0; i < len; i++)
		{
			add(t);
		}
	}

	void unit(int width)
	{
		int w, h, c, v;
		reShape({ width,width,shape.channels,shape.variable });
		tensor.clear();
		for (v = 0; v < shape.variable; v++)
		{
			for (c = 0; c < shape.channels; c++)
			{
				for (h = 0; h < shape.height; h++)
				{
					for (w = 0; w < shape.width; w++)
					{
						if (w == h)
						{
							add(1);
						}
						else
						{
							add(0);
						}
					}
				}
			}
		}
	}

	bool insert(int pos, double value)
	{
		if (pos > tensor.size())
		{
			return false;
		}
		tensor.insert(tensor.begin() + pos, value);
		return true;
	}

	bool insert(int col, int row, int channel, int batch, double value)
	{
		int pos = col + row * shape.width + channel * shape.height * shape.width + batch * shape.channels * shape.height * shape.width;
		if ((unsigned int)pos > tensor.size())
		{
			return false;
		}
		tensor.insert(tensor.begin() + pos, value);
		return true;
	}

	void padding2d(int col, int row)
	{
		int i, j, c, r;
		if (row >= 0)shape.height++;
		for (i = 0; i < shape.variable; i++)
		{
			for (j = 0; j < shape.channels; j++)
			{
				for (c = 0; c < shape.width && row >= 0; c++)
				{
					insert(c, row, j, i, 0);
				}		
			}
		}
		if (col >= 0)shape.width++;
		for (i = 0; i < shape.variable; i++)
		{
			for (j = 0; j < shape.channels; j++)
			{
				for (r = 0; r < shape.height && col >= 0; r++)
				{
					insert(col, r, j, i, 0);
				}
			}
		}
	}

	void channelsMerge(int i, int j)
	{
		int w, h, b;
		for (b = 0; b < shape.variable; b++)
		{
			for (h = 0; h < shape.height; h++)
			{
				for (w = 0; w < shape.width; w++)
				{
					set(w, h, i, b, get(w, h, i, b) + get(w, h, j, b));
				}
			}
		}
		shape.channels--;
		int len = shape.width*shape.height;
		for (b = 0; b < shape.variable; b++)
		{
			/*
			for (h = 0; h < shape.height; h++)
			{
				for (w = 0; w < shape.width; w++)
				{
					remove(0, 0, j, b);
				}
			}*/
			remove(b*shape.channels*len + j * len, b*shape.channels*len + (j + 1) * len);
		}
	}

	void channelsCopy(int channels)
	{
		int w, h, v;
		for (v = 0; v < shape.variable; v++)
		{
			for (h = 0; h < shape.height; h++)
			{
				for (w = 0; w < shape.width; w++)
				{
					insert(w, h, channels + 1, v, get(w, h, channels, v));
				}
			}
		}
		shape.channels++;
	}

	void variableMerge(int i,int j)
	{
		int w, h, c;
		for (c = 0; c < shape.channels; c++)
		{
			for (h = 0; h < shape.height; h++)
			{
				for (w = 0; w < shape.width; w++)
				{
					set(w, h, c, i, get(w, h, c, i) + get(w, h, c, j));
				}
			}
		}
		shape.variable--;
		/*
		for (c = 0; c < shape.channels; c++)
		{
			for (h = 0; h < shape.height; h++)
			{
				for (w = 0; w < shape.width; w++)
				{
					remove(0, 0, 0, j);
				}
			}
		}*/
		int len = shape.width*shape.height*shape.channels;
		remove(j*len, (j + 1)*len);
	}

	void variableCopy(int variable)
	{
		int w, h, c;
		for (c = 0; c < shape.channels; c++)
		{
			for (h = 0; h < shape.height; h++)
			{
				for (w = 0; w < shape.width; w++)
				{
					insert(w, h, c, variable+1, get(w, h, c, variable));
				}
			}
		}
		shape.variable++;
	}

	void variableSplice(Tensor t)
	{
		if (shape.width != t.shape.width || shape.height != t.shape.height || shape.channels != t.shape.channels)
		{
			return;
		}
		int w, h, c, v;
		for (v = 0; v < t.shape.variable; v++)
		{
			for (c = 0; c < t.shape.channels; c++)
			{
				for (h = 0; h < t.shape.height; h++)
				{
					for (w = 0; w < t.shape.width; w++)
					{
						tensor.push_back(t.get(w, h, c, v));
					}
				}
			}
		}
		shape.variable += t.shape.variable;
	}

	void nextBatch()
	{
		if (batch > 1)
		{
			int len = shape.width*shape.height*shape.channels;
			for (size_t i = 0; i < len; i++)
			{
				remove(0, 0, 0, 0);
			}
			batch--;
		}
	}

	void replaceBatch(double data[], int size, int batchs)
	{
		tensor.clear();
		for (size_t i = 0; i < size; i++)
		{
			tensor.push_back(data[i]);
		}
		shape.variable = batchs;
		batch = batchs;
	}

	void flip2d()
	{
		int c, v;
		int len = shape.width*shape.height;
		int i = 0;
		int p = 0;
		int w = shape.width, h = shape.height;
		double t;
		for (v = 0; v < shape.variable; v++)
		{
			for (c = 0; c < shape.channels; c++)
			{
				for (i = 0; i < len; i++)
				{
					p = (i%w)*h + i / w;
					while (p > i)
					{
						p = (p%w)*h + p / w;
					}
					if (p == i)
					{
						t = get(i%w, i / w, c, v);
						int n = (p%h)*w + p / h;//(p%w)*h + (p / w);//
						while (n != i)
						{
							set(p%w, p / w, c, v, get(n%w, n / w, c, v));
							p = n;
							n = (p%h)*w + (p / h);
						}
						set(p%w, p / w, c, v, t);
					}
				}
			}
		}
		v = shape.width;
		shape.width = shape.height;
		shape.height = v;	
	}

	void rota2D()
	{
		int len = shape.width*shape.height;
		int exchangeLen = ceil(len / 2.0);
		double t = 0;
		int pos;
		for (size_t v = 0; v < shape.variable; v++)
		{
			for (size_t c = 0; c < shape.channels; c++)
			{
				pos = c * len + v * shape.channels*len;
				for (size_t l = 0; l < exchangeLen; l++)
				{
					t = get(l + pos);
					set(l + pos, get(len - l - 1 + pos));
					set(len - l - 1 + pos, t);
				}
			}
		}
	}

	friend Tensor operator+(const Tensor t1, const Tensor t2)
	{
	}

	friend Tensor operator*(const Tensor t1, const Tensor t2)
	{
	}

	friend std::ostream& operator<<(std::ostream &output, const Tensor &t)
	{
		int w, h, c, b;
		output << std::endl;
		for (b = 0; b < t.shape.variable; b++)
		{
			output << "{ " << std::endl;
			for (c = 0; c < t.shape.channels; c++)
			{
				output << "[ " << std::endl;
				for (h = 0; h < t.shape.height; h++)
				{
					output << "[ ";
					for (w = 0; w < t.shape.width; w++)
					{
						output << t.tensor[w + h * t.shape.width + c * t.shape.height * t.shape.width + b * t.shape.channels * t.shape.height * t.shape.width] << " ";
					}
					output << "]" << std::endl;
				}
				output << "]" << std::endl;
			}
			output << "}" << std::endl;
		}
		output << std::endl;
		return output;
	}

	extern friend class Graph;

	//ArcNode

	void setNodeID(int id)
	{
		nodeid = id;
	}

	void setKind(VertexKind kind)
	{
		vkind = kind;
	}

	int getNodeID()
	{
		return nodeid;
	}

	Parameter* getPar()
	{
		return par;
	}

private:


private:
	std::vector<double> tensor;
	Shape shape;
	int length;
	int batch;

	int nodeid;
	ArcNode *firstarc;
	ArcNode *inversearc;
	VertexKind vkind;

	Parameter *par;

};