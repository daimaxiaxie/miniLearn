#pragma once

#include "head.h"

template <typename T>
T average(int n, T type, ...)
{
	if (n < 2)
	{
		return type;
	}
	va_list ap;
	__crt_va_start(ap, type);
	T temp = type;
	for (size_t i = 1; i < (unsigned int)n; i++)
	{
		temp += __crt_va_arg(ap, T);
	}
	temp = temp / n;
	__crt_va_end(ap);
	return temp;
}

template <typename T>
T max(int n, T type, T type1, ...)
{
	if (n < 2)
	{
		return type > type1 ? type : type1;
	}
	va_list ap;
	__crt_va_start(ap, type);
	T temp = type;
	for (size_t i = 1; i < (unsigned int)n; i++)
	{
		T t = __crt_va_arg(ap, T);
		if (temp < t)
		{
			temp = t;
		}
	}
	//va_end(ap);
	__crt_va_end(ap);
	return temp;
}

template <typename T>
T max(T t1, T t2)
{
	return t1 > t2 ? t1 : t2;
}

double max(int n, double t[])
{
	double temp = t[0];
	for (size_t i = 1; i < (unsigned int)n; i++)
	{
		if (temp < t[i])
		{
			temp = t[i];
		}
	}
	return temp;
}

double rand_back(double i, double j)
{
	double u1 = double(rand() % 1000) / 1000, u2 = double(rand() % 1000) / 1000, r;
	static unsigned int seed = 0;
	r = i + sqrt(j)*sqrt(-2.0*(log(u1) / log(exp(1.0))))*cos(2 * 3.1415926*u2);
	return r;
}
