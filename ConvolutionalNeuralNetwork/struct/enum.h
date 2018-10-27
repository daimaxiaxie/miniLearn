#pragma once

enum padding
{
	SAME,
	VALID,
};

enum VertexKind
{
	Variable,
	Constant,
	//Operator,
	Addition,                  // +
	Subtraction,               // -
	Multiplication,            // *
	Hadamard,                  // กั
	Division,                  // /
	Exp,                       // exp
	Logarithm,                 // log
	Neg,                       // -
	Abs,                       // | |
	Square,                    // 2

	Reduce_sum,
	ReShape,

	Convolution,
	MaxPooling,
	AveragePooling,
	Relu,
	Dropout,
	Softmax
};

enum InfoType
{
	input_1,
	input_2
};

struct ArcNode
{
	int adjvex;
	ArcNode *next;
	InfoType info;
};
