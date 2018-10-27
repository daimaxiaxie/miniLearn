#pragma once

#include "head.h"

class csv
{
private:
	std::string s[1000];
	std::string a;
	std::ifstream fin;
public:
	csv()
	{

	}
	csv(std::string aa) :a(aa)
	{
		fin.open(a, std::ios::in);
		if (!fin)
		{
			std::cout << "open error" << std::endl;
			exit(0);
		}
	}
	~csv() {
		fin.close();
	}

	void setURL(std::string u)
	{
		a = u;
		fin.open(u, std::ios::in);
		if (!fin)
		{
			std::cout << "open error" << std::endl;
			system("pause");
			exit(0);
		}
	}

	void csv_read()
	{
		int i = 0;
		char al;
		for (int j = 0; j < 1000; j++)
		{
			s[j] = "";
		}
		while (i < 1000)
		{
			fin.get(al);
			if (al != ','&&al != '\n')
			{
				s[i] += al;
			}
			else if (al == '\n')
			{
				break;
			}
			else
			{
				i++;
			}
		}
	}

	std::string* csv_get()
	{
		return s;
	}
};

class mnist_train
{
public:
	mnist_train()
	{
		data.setURL(url);
	}

	~mnist_train()
	{
		//data.~csv();
	}

	void readline()
	{
		data.csv_read();
		d = data.csv_get();
	}

	double get(int x)
	{
		return stringToNum(d[x]);
	}

	void getbatch(double **y, double **x, int batch)
	{
		for (size_t i = 0; i < batch; i++)
		{
			readline();
			for (size_t j = 0; j < 785; j++)
			{
				if (j == 0)
				{
					int k = get(0);
					y[i][k] = 1.0;
				}
				else
				{
					x[i][j - 1] = get(i);
				}
			}
		}
	}

	void getbatch(double *y, double *x, int batch)
	{
		for (size_t i = 0; i < batch; i++)
		{
			readline();
			for (size_t j = 0; j < 785; j++)
			{
				if (j == 0)
				{
					int k = get(0);
					y[k + i * 10] = 1.0;
				}
				else
				{
					x[(j - 1) + i * 784] = get(j);
				}
			}
		}
	}

private:
	double stringToNum(std::string& str)
	{
		std::istringstream iss(str);
		double num;
		iss >> num;
		return num;
	}

private:
	std::string *d;
	std::string url = "train.csv";
	csv data;
};

class mnist_test
{
public:
	mnist_test()
	{
		data.setURL(url);
	}

	~mnist_test()
	{
		//data.~csv();
	}

	void readline()
	{
		data.csv_read();
		d = data.csv_get();
	}

	double get(int x)
	{
		return stringToNum(d[x]);
	}

	void getbatch(double *y, double *x, int batch)
	{
		for (size_t i = 0; i < batch; i++)
		{
			readline();
			for (size_t j = 0; j < 785; j++)
			{
				if (j == 0)
				{
					int k = get(0);
					y[k + i * 10] = 1.0;
				}
				else
				{
					x[(j - 1) + i * 784] = get(j);
				}
			}
		}
	}

private:
	double stringToNum(std::string& str)
	{
		std::istringstream iss(str);
		double num;
		iss >> num;
		return num;
	}

private:
	std::string *d;
	std::string url = "test.csv";
	csv data;
};