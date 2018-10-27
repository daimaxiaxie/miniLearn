 #pragma once

#include "CNN.h"

class Graph
{
public:
	Graph()
	{
		vertex.clear();
		vexnum = 0;
		arcnum = 0;
		istrain = false;
	}
	~Graph()
	{
		for (size_t i = 0; i < vertex.size(); i++)
		{
			while (vertex[i].firstarc != nullptr&&vertex[i].inversearc != nullptr)
			{
				ArcNode *s;
				s=vertex[i].firstarc;
				vertex[i].firstarc = s->next;
				ArcNode *p;
				p = vertex[i].inversearc;
				vertex[i].inversearc = p->next;
				delete s, p;
			}
		}
		vertex.clear();
	}

	Tensor& addVertex(Tensor *tensor)
	{
		tensor->setNodeID(vertex.size());
		vertex.push_back(*tensor);
		vexnum++;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor &tensor)
	{
		tensor.setNodeID(vertex.size());
		vertex.push_back(tensor);
		vexnum++;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor *tensor, Tensor *input, Tensor *filter, Shape strides, padding pad)
	{
		tensor->setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor->nodeid;
		s->next = input->firstarc;
		s->info = input_1;
		input->firstarc = s;
		ArcNode *p = new ArcNode();
		p->adjvex = tensor->nodeid;
		p->next = filter->firstarc;
		p->info = input_2;
		filter->firstarc = p;
		tensor->par = new Parameter();
		tensor->par->strides = strides;
		tensor->par->pad = pad;
		ArcNode *t1 = new ArcNode();
		ArcNode *t2 = new ArcNode();
		t1->adjvex = input->nodeid;
		t1->next = tensor->inversearc;
		t1->info = input_1;
		tensor->inversearc = t1;
		t2->adjvex = filter->nodeid;
		t2->next = tensor->inversearc;
		t2->info = input_2;
		tensor->inversearc = t2;
		vertex.push_back(*tensor);
		vexnum++;
		arcnum += 2;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor tensor,Tensor input, Tensor filter, Shape strides, padding pad)
	{
		tensor.setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor.nodeid;
		s->next = vertex[input.nodeid].firstarc;
		s->info = input_1;
		vertex[input.nodeid].firstarc = s;
		ArcNode *p = new ArcNode();
		p->adjvex = tensor.nodeid;
		p->next = vertex[filter.nodeid].firstarc;
		p->info = input_2;
		vertex[filter.nodeid].firstarc = p;
		tensor.par = new Parameter();
		tensor.par->strides = strides;
		tensor.par->pad = pad;
		ArcNode *t1 = new ArcNode();
		ArcNode *t2 = new ArcNode();
		t1->adjvex = vertex[input.nodeid].nodeid;
		t1->next = tensor.inversearc;
		t1->info = input_1;

		tensor.inversearc = t1;
		t2->adjvex = vertex[filter.nodeid].nodeid;
		t2->next = tensor.inversearc;
		t2->info = input_2;

		tensor.inversearc = t2;
		vertex.push_back(tensor);
		vexnum++;
		arcnum += 2;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor tensor, Tensor input, Shape ksize, Shape strides, padding pad)
	{
		tensor.setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor.nodeid;
		s->next = vertex[input.nodeid].firstarc;
		s->info = input_1;
		vertex[input.nodeid].firstarc = s;
		tensor.par = new Parameter();
		tensor.par->ksize = ksize;
		tensor.par->strides = strides;
		tensor.par->pad = pad;
		ArcNode *t1 = new ArcNode();
		t1->adjvex = vertex[input.nodeid].nodeid;
		t1->next = tensor.inversearc;
		t1->info = input_1;
		tensor.inversearc = t1;
		vertex.push_back(tensor);
		vexnum++;
		arcnum++;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor tensor, Tensor input)
	{
		tensor.setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor.nodeid;
		s->next = vertex[input.nodeid].firstarc;
		s->info = input_1;
		vertex[input.nodeid].firstarc = s;
		ArcNode *t = new ArcNode();
		t->adjvex = input.nodeid;
		t->next = tensor.inversearc;
		t->info = input_1;
		tensor.inversearc = t;
		vertex.push_back(tensor);
		vexnum++;
		arcnum++;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor tensor, Tensor input, float keep_prob)
	{
		tensor.setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor.nodeid;
		s->next = vertex[input.nodeid].firstarc;
		s->info = input_1;
		vertex[input.nodeid].firstarc = s;
		ArcNode *t = new ArcNode();
		t->adjvex = input.nodeid;
		t->next = tensor.inversearc;
		t->info = input_1;
		tensor.inversearc = t;
		tensor.par = new Parameter();
		tensor.par->keep_prob = keep_prob;

		vertex.push_back(tensor);
		vexnum++;
		arcnum++;
		return vertex[vertex.size() - 1];
	}


	Tensor& addVertex(Tensor tensor, Tensor input, int axis)
	{
		tensor.setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor.nodeid;
		s->next = vertex[input.nodeid].firstarc;
		s->info = input_1;
		vertex[input.nodeid].firstarc = s;
		ArcNode *t = new ArcNode();
		t->adjvex = input.nodeid;
		t->next = tensor.inversearc;
		t->info = input_1;
		tensor.inversearc = t;
		tensor.par = new Parameter();
		tensor.par->axis = axis;

		vertex.push_back(tensor);
		vexnum++;
		arcnum++;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor tensor, Tensor input, Shape shape)
	{
		tensor.setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor.nodeid;
		s->next = vertex[input.nodeid].firstarc;
		s->info = input_1;
		vertex[input.nodeid].firstarc = s;
		ArcNode *t = new ArcNode();
		t->adjvex = input.nodeid;
		t->next = tensor.inversearc;
		t->info = input_1;
		tensor.inversearc = t;
		tensor.par = new Parameter();
		tensor.par->shape = input.getShape();
		tensor.par->newshape = shape;
		vertex.push_back(tensor);
		vexnum++;
		arcnum++;
		return vertex[vertex.size() - 1];
	}

	Tensor& addVertex(Tensor tensor, Tensor input1, Tensor input2)
	{
		tensor.setNodeID(vertex.size());
		ArcNode *s = new ArcNode();
		s->adjvex = tensor.nodeid;
		s->next = vertex[input1.nodeid].firstarc;
		s->info = input_1;
		vertex[input1.nodeid].firstarc = s;
		ArcNode *p = new ArcNode();
		p->adjvex = tensor.nodeid;
		p->next = vertex[input2.nodeid].firstarc;
		p->info = input_2;
		vertex[input2.nodeid].firstarc = p;

		ArcNode *t1 = new ArcNode();
		ArcNode *t2 = new ArcNode();
		t1->adjvex = vertex[input1.nodeid].nodeid;
		t1->next = tensor.inversearc;
		t1->info = input_1;
		tensor.inversearc = t1;
		t2->adjvex = vertex[input2.nodeid].nodeid;
		t2->next = tensor.inversearc;
		t2->info = input_2;
		tensor.inversearc = t2;
		vertex.push_back(tensor);
		vexnum++;
		arcnum += 2;
		return vertex[vertex.size() - 1];
	}

	Tensor& getVertex(int i)
	{
		return vertex[i];
	}

	Tensor& run(Tensor tensor)
	{
		istrain = false;
		return ForwardPropagation(tensor.nodeid);
	}

public:
	class train
	{
	public:
		train()
		{
		}

		void GradientDescent(float learning_rate, Tensor minimize)
		{
			graph->istrain = true;
			gradient.clear();
			int batch = getBatch(minimize.nodeid);
			int batch_num = batch;
			do
			{
				Tensor residual_last = Tensor(0, graph->vertex[minimize.nodeid].shape);

				std::cout << "\r" << (float)(batch_num - batch) / batch_num * 100 << " %";
				
				while (true)
				{
					Tensor &residual = graph->ForwardPropagation(minimize.nodeid);
					if (residual_last.shape != residual.shape)
					{
						residual_last.fill(0);
					}
					if (isEnd(residual_last, graph->vertex[minimize.nodeid]))
					{
						break;
					}
					
					gradient = graph->vertex;
					BackPropagation_Grad(residual.nodeid);
					//std::cout << graph->vertex[29];

					//std::cout << graph->vertex[25];

					ApplyGrad(residual.nodeid, learning_rate);
					
					residual_last = graph->vertex[minimize.nodeid];
				};

				if (--batch < 1)
				{
					break;
				}
				nextBatch(minimize.nodeid);
			} while (true);
			std::cout << "\r" << "Train Complete" << std::endl;
		}

	private:

		Tensor grad(Tensor t1, Tensor t2)
		{
			VertexKind Kind = t1.vkind;
			if (t1.inversearc == nullptr)
			{
				return Tensor({ 0,0,0,0 });
			}
			else if (Kind == Addition)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (w == h)
								{
									tensor.add(1);
								}
								else
								{
									tensor.add(0);
								}
								
							}
						}
					}
				}
				return tensor;
			}
			else if (Kind == Subtraction)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				if (t1.inversearc->adjvex == t2.nodeid&&t1.inversearc->info == input_2)
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t1.getShape().channels; c++)
						{
							for (h = 0; h < tensor.shape.height; h++)
							{
								for (w = 0; w < tensor.shape.width; w++)
								{
									if (w == h)
									{
										tensor.add(-1);
									}
									else
									{
										tensor.add(0);
									}

								}
							}
						}
					}
				}
				else if (t1.inversearc->next->adjvex == t2.nodeid&&t1.inversearc->next->info == input_1)
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t1.getShape().channels; c++)
						{
							for (h = 0; h < tensor.shape.height; h++)
							{
								for (w = 0; w < tensor.shape.width; w++)
								{
									if (w == h)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}

								}
							}
						}
					}
				}
				return tensor;
			}
			else if (Kind == Multiplication)
			{
				int width, height;
				width = t2.shape.width*t2.getShape().height;
				height = t1.getShape().width*t1.getShape().height;
				Tensor tensor = Tensor({ width,height,t1.getShape().channels,t1.getShape().variable });
				int w, h, c, v, m, n;
				m = 0;
				n = 0;
				if (t1.inversearc->adjvex == t2.nodeid&&t1.inversearc->info == input_2)
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t1.getShape().channels; c++)
						{
							for (h = 0; h < height; h++)
							{
								for (w = 0; w < width; w++)
								{
									int n, h1;
									n = t2.getShape().width;
									h1 = h % n;
									if (w % n == h1)
									{
										tensor.add(graph->vertex[t1.inversearc->next->adjvex].get(w / n, h / n, c, v));
									}
									else
									{
										tensor.add(0);
									}
								}
							}
						}
					}
				}
				else if (t1.inversearc->next->adjvex == t2.nodeid&&t1.inversearc->next->info == input_1)
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t1.getShape().channels; c++)
						{
							for (h = 0; h < height; h++)
							{
								int h1 = h / t1.shape.width;
                                int l1, l2, l3;
							    l1 = t2.getShape().width*h1;
								l2 = t2.getShape().width;
								l3 = width - l2 - l1;
								for (w = 0; w < width; w++)
								{
									if (w >= l1 && w < l1 + l2)
									{
										tensor.add(graph->vertex[t1.inversearc->adjvex].get(h%t1.shape.width, w - l1, c, v));
									}
									else
									{
										tensor.add(0);
									}
								}
							}
						}
					}
				}

				/*
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (w = 0; w < tensor.shape.width; w++)
						{
							double t = 0;
							for (h = 0; h < tensor.shape.height; h++)
							{
								t += tensor.get(w, h, c, v);
							}
							tensor.set(w, 0, c, v, t);
						}
					}
				}
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								tensor.remove(0, 1, c, v);
							}
						}
					}
				}
				*/
				/*
				tensor=reduce_sum(tensor, 1);
				tensor.reShape(t2.getShape());
				*/
				return tensor;
			}
			else if (Kind == Hadamard)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				if (t1.inversearc->adjvex == t2.nodeid&&t1.inversearc->info == input_2)
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t1.getShape().channels; c++)
						{
							for (h = 0; h < tensor.shape.height; h++)
							{
								for (w = 0; w < tensor.shape.width; w++)
								{
									if (w == h)
									{
										tensor.add(graph->vertex[t1.inversearc->next->adjvex].get(h));
									}
									else
									{
										tensor.add(0);
									}

								}
							}
						}
					}
				}
				else if (t1.inversearc->next->adjvex == t2.nodeid&&t1.inversearc->next->info == input_1)
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t1.getShape().channels; c++)
						{
							for (h = 0; h < tensor.shape.height; h++)
							{
								for (w = 0; w < tensor.shape.width; w++)
								{
									if (w == h)
									{
										tensor.add(graph->vertex[t1.inversearc->adjvex].get(h));
									}
									else
									{
										tensor.add(0);
									}

								}
							}
						}
					}
				}
				return tensor;
			}
			else if (Kind == Exp)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (w == h)
								{
									tensor.add(t1.get(w));
								}
								else
								{
									tensor.add(0);
								}

							}
						}
					}
				}
				return tensor;
            }
			else if (Kind == Logarithm)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (w == h)
								{
									tensor.add(1 / t2.get(w));
								}
								else
								{
									tensor.add(0);
								}
							}
						}
					}
				}
				return tensor;
            }
			else if (Kind == Neg)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (w == h)
								{
									tensor.add(-1);
								}
								else
								{
									tensor.add(0);
								}

							}
						}
					}
				}
				return tensor;
            }
			else if (Kind == Abs)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (w == h)
								{
									if (t2.get(w) < 0)
									{
										tensor.add(-1);
									}
									else if (t2.get(w) > 0)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}
								}
								else
								{
									tensor.add(0);
								}
							}
						}
					}
				}
				return tensor;
			}
			else if (Kind == Square)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (w == h)
								{
									tensor.add(2 * t2.get(w));
								}
								else
								{
									tensor.add(0);
								}

							}
						}
					}
				}
				return tensor;
            }
			else if (Kind == Reduce_sum)
			{
			    Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t2.shape.channels,t2.shape.variable });
				int w, h, c, v;
				int axis = t1.par->axis;
				for (v = 0; v < t2.getShape().variable; v++)
				{
					for (c = 0; c < t2.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							int n = t2.shape.width;
							int n1 = n * h;
							int n2 = n1 + n;
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (axis == 0)
								{
									if (w >= n1&&w < n2)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}
								}
								else if(axis==1)
								{
									n1 = n + h;
									if (w%n == h)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}
								}
								else if (axis == 2)
								{
									if (w == h)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}
								}
								else if (axis == 3)
								{
									if (w == h)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}
									//This is an error
								}
							}
						}
					}
				}
				return tensor;
            }
			else if (Kind == Softmax)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t2.shape.channels,t2.shape.variable });
				int w, h, c, v;
				if (t2.shape.channels == 1)
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t2.getShape().channels; c++)
						{
							for (h = 0; h < tensor.shape.height; h++)
							{
								for (w = 0; w < tensor.shape.width; w++)
								{
									int h1 = h / t1.shape.width;
									int l1, l2;
									l1 = h1 * t1.shape.width;
									l2 = l1 + t1.shape.width;
									if (w >= l1 && w < l2)
									{
										if (w == h)
										{
											tensor.add(t1.get(h)*(1 - t1.get(h)));
										}
										else
										{
											tensor.add(0 - t1.get(h)*t1.get(w));
										}
									}
									else
									{
										tensor.add(0);
									}

								}
							}
						}
					}
				}//std::cout << tensor;
				/*
				else
				{
					for (v = 0; v < t1.getShape().variable; v++)
					{
						for (c = 0; c < t1.getShape().channels; c++)
						{
							for (h = 0; h < tensor.shape.height; h++)
							{
								for (w = 0; w < tensor.shape.width; w++)
								{
									if (w == h)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}

								}
							}
						}
					}
				}
                */
				return tensor;
			}
			else if (Kind == Relu)
			{
				Tensor tensor = Tensor({ t2.shape.width*t2.shape.height,t1.shape.width*t1.shape.height,t1.shape.channels,t1.shape.variable });
				int w, h, c, v;
				for (v = 0; v < t1.getShape().variable; v++)
				{
					for (c = 0; c < t1.getShape().channels; c++)
					{
						for (h = 0; h < tensor.shape.height; h++)
						{
							for (w = 0; w < tensor.shape.width; w++)
							{
								if (w == h)
								{
									if (t1.get(w) > 0)
									{
										tensor.add(1);
									}
									else
									{
										tensor.add(0);
									}
								}
								else
								{
									tensor.add(0);
								}

							}
						}
					}
				}
				return tensor;
			}
		}

		bool Convergence(int index)
		{
			return false;
		}

		void AutoDiff(int index, Tensor grad, Tensor *result, int end)
		{
			if (index == end)
			{
				grad = reduce_sum(grad, 1);
				grad.reShape(graph->vertex[index].shape);
				*result = add(*result, grad);
				return;
			}
			VertexKind Kind = graph->vertex[index].vkind;
			if (graph->vertex[index].inversearc == nullptr)
			{
				//std::cout << index << std::endl;
				grad=reduce_sum(grad, 1);
				grad.reShape(graph->vertex[index].shape);
				if (Kind == Variable)
				{
					//std::cout << grad;
				}
				else if (Kind == Constant)
				{
					//std::cout << grad;
				}
				//return graph->vertex[index];
				return;
			}
			if ( index< 0)
			{
			}
			else if (Kind == Reduce_sum)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
			}
			else if (Kind == Addition)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
				AutoDiff(graph->vertex[index].inversearc->next->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->next->adjvex])), result, end);
			}
			else if (Kind == Subtraction)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
				AutoDiff(graph->vertex[index].inversearc->next->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->next->adjvex])), result, end);
			}
			else if (Kind == Multiplication)
			{
				//std::cout << grad;
				//std::cout << train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex]);
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
				AutoDiff(graph->vertex[index].inversearc->next->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->next->adjvex])), result, end);
			}
			else if (Kind == Division)
			{

			}
			else if (Kind == Neg)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
			}
			else if (Kind == Abs)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
			}
			else if (Kind == Exp)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
			}
			else if (Kind == Logarithm)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
			}
			else if (Kind == Hadamard)
			{
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
				AutoDiff(graph->vertex[index].inversearc->next->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->next->adjvex])), result, end);
			}
			else if (Kind == ReShape)
			{
				//loss_delta.reShape(grad.getShape());
				//loss_delta = add(loss_delta, grad);
				//return;
				AutoDiff(graph->vertex[index].inversearc->adjvex, grad, result, end);
			}
			else if (Kind == Softmax)
			{
				//*result = add(*result, grad);
				//return;
				//std::cout << grad;
				//std::cout << train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex]);
				//std::cout << matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex]));
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
			}
			else if (Kind == Relu)
			{
				//*result = add(*result, grad);
				//return;
				AutoDiff(graph->vertex[index].inversearc->adjvex, matmul(grad, train::grad(graph->vertex[index], graph->vertex[graph->vertex[index].inversearc->adjvex])), result, end);
			}
		}

		void BackPropagation_Grad(int index)
		{
			Tensor unit = Tensor({ 1,1,1,1 });                                                      //code chaos
			unit.unit(graph->vertex[index].shape.width*graph->vertex[index].shape.height);
			int active_index = getNextLevel(index);
			if (active_index < 0)
			{
				return;
			}
			Tensor loss_grad = Tensor(graph->vertex[active_index].shape);
			loss_grad.fill_clear(0);
			AutoDiff(index, unit, &loss_grad, graph->vertex[active_index].inversearc->adjvex);
			Tensor active_grad = grad(graph->vertex[active_index], graph->vertex[graph->vertex[active_index].inversearc->adjvex]);
			//std::cout << loss_grad;
			//gradient[1].flip2d();
			//std::cout << matmul(loss_grad, gradient[1]);
			//return;
			//std::cout << active_grad;
			active_grad = reduce_sum(active_grad, 1);
			active_grad.reShape(graph->vertex[graph->vertex[active_index].inversearc->adjvex].shape);
			//std::cout << active_grad;
			//Tensor delta = hadamard(loss_grad, active_grad);
			int id = graph->vertex[active_index].inversearc->adjvex;
			
			//loss_grad.fill_clear(0.10000000149011611938);
			//loss_grad.set(7, -0.90000003576278686523);
			Tensor delta = loss_grad;
			//std::cout << delta;
			//return;
			//delta = unit;
			//id = 37;
			while (id > 0)
			{
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout)
				{
					//id = getNextLevel(graph->vertex[id].inversearc->adjvex);
					if (id < 0)
					{
						return;
					}
					else if (kind == MaxPooling)
					{
						//if (id> 10)delta.fill_clear(0.00000019073505086);
						delta = inverse_maxpool(delta, graph->vertex[id], graph->vertex[graph->vertex[id].inversearc->adjvex]);
						kind = graph->vertex[graph->vertex[id].inversearc->adjvex].vkind;
						if (kind == ReShape)
						{
							id = graph->vertex[graph->vertex[id].inversearc->adjvex].inversearc->adjvex;
						}
						else
						{
							id = graph->vertex[id].inversearc->adjvex;
						}
						kind = graph->vertex[id].vkind;
						if (kind == Relu)
						{
							active_grad = grad(graph->vertex[id], graph->vertex[graph->vertex[id].inversearc->adjvex]);
							active_grad = reduce_sum(active_grad, 1);
							active_grad.reShape(graph->vertex[id].shape);
							delta = hadamard(delta, active_grad);
							id = graph->vertex[id].inversearc->adjvex;
						}
						
					}
					else if (kind == AveragePooling)
					{
						delta = inverse_avgpool(delta, graph->vertex[id], graph->vertex[graph->vertex[id].inversearc->adjvex]);
						kind = graph->vertex[graph->vertex[id].inversearc->adjvex].vkind;
						if (kind == ReShape)
						{
							id = graph->vertex[graph->vertex[id].inversearc->adjvex].inversearc->adjvex;
						}
						else
						{
							id = graph->vertex[id].inversearc->adjvex;
						}
						kind = graph->vertex[id].vkind;
						if (kind == Relu)
						{
							active_grad = grad(graph->vertex[id], graph->vertex[graph->vertex[id].inversearc->adjvex]);
							active_grad = reduce_sum(active_grad, 1);
							active_grad.reShape(graph->vertex[id].shape);
							delta = hadamard(delta, active_grad);
							id = graph->vertex[id].inversearc->adjvex;
						}
					}
					else if (kind == Convolution)
					{
						int w_index = getInput(id, input_2);
						int i_index = getInput(id, input_1);
						Tensor weight = graph->vertex[w_index];
						//gradient[w_index] = conv2d(graph->vertex[i_index], delta, graph->vertex[id].par->strides, graph->vertex[id].par->pad);
						gradient[w_index] = inverse_conv_weight(weight, graph->vertex[i_index], delta, graph->vertex[id].par->strides, graph->vertex[id].par->pad);
						//std::cout << gradient[w_index]; return;
						//weight.flip2d();
						//std::cout << matmul(delta, graph->vertex[i_index]); return;
						//delta = conv2d(delta, weight, graph->vertex[id].par->strides, graph->vertex[id].par->pad);
						//std::cout << delta; //return;
						delta = inverse_conv(graph->vertex[i_index], delta, weight, graph->vertex[id].par->strides);
						
						kind = graph->vertex[i_index].vkind;
						if (kind == ReShape)
						{
							id = graph->vertex[i_index].inversearc->adjvex;
						}
						else
						{
							id = i_index;
						}
						if (id < 0)
						{
							return;
						}
						kind = graph->vertex[id].vkind;
						if (kind == Relu)
						{
							active_grad = grad(graph->vertex[id], graph->vertex[graph->vertex[id].inversearc->adjvex]);
							active_grad = reduce_sum(active_grad, 1);
							active_grad.reShape(graph->vertex[id].shape);
							delta = hadamard(delta, active_grad);
							id = graph->vertex[id].inversearc->adjvex;
						}
						
					}
					else if (kind == Dropout)
					{
						delta = inverse_dropout(graph->vertex[graph->vertex[id].inversearc->adjvex], graph->vertex[id], delta);
						kind = graph->vertex[graph->vertex[id].inversearc->adjvex].vkind;
						if (kind == ReShape)
						{
							id = graph->vertex[graph->vertex[id].inversearc->adjvex].inversearc->adjvex;
						}
						else
						{
							id = graph->vertex[id].inversearc->adjvex;
						}
						kind = graph->vertex[id].vkind;
						if (kind == Relu)
						{
							active_grad = grad(graph->vertex[id], graph->vertex[graph->vertex[id].inversearc->adjvex]);
							active_grad = reduce_sum(active_grad, 1);
							active_grad.reShape(graph->vertex[id].shape);
							delta = hadamard(delta, active_grad);
							id = graph->vertex[id].inversearc->adjvex;
						}
					}
				}
				else if (kind == ReShape)
				{
					id = graph->vertex[id].inversearc->adjvex;
					delta.reShape(graph->vertex[id].shape);
				}
				else if (kind == Constant)
				{
					id = -1;
				}
				else
				{
					if (kind == Addition)
					{
						int _index = getInput(id, input_2);
						if (graph->vertex[_index].vkind == Convolution)
						{
							_index = getInput(id, input_1);
							gradient[_index] = reduce_sum(delta, 0);
							gradient[_index] = reduce_sum(gradient[_index], 1);
							id = getInput(id, input_2);
							continue;
						}
						else
						{
							_index = getInput(id, input_1);
						}
						if (graph->vertex[_index].vkind == Convolution)
						{
							_index = getInput(id, input_2);
							gradient[_index] = reduce_sum(delta, 0);
							gradient[_index] = reduce_sum(gradient[_index], 1);
							id = getInput(id, input_1);
							continue;
						}
					}
					int w_index = getWeight(id);
					int b_index = getBias(id);
					int i_index = getInput(id);
					if (w_index > -1 || b_index > -1)
					{
						if (graph->vertex[id].vkind == ReShape)
						{
							delta.reShape(graph->vertex[id].par->shape);
							id = graph->vertex[id].inversearc->adjvex;
						}
						Tensor weight = graph->vertex[w_index];
						gradient[b_index] = delta;
						gradient[i_index].flip2d();
						bool isWX = judgeWX(id);
						if (isWX)
						{
							gradient[w_index] = matmul(delta, gradient[i_index]);
						}
						else
						{
							gradient[w_index] = matmul(gradient[i_index], delta);
						}

						
						id = getNextLevel(id);
						//std::cout << delta;
						//return;
						if (id < 0)
						{
							return;
						}
						kind = graph->vertex[id].vkind;
						/*
						if (kind == Dropout)
						{
							id = getNextLevel(graph->vertex[id].inversearc->adjvex);
						}*/
						if (id < 0)
						{
							return;
						}
						if (kind == Softmax || kind == Relu)
						{
							active_grad = grad(graph->vertex[id], graph->vertex[graph->vertex[id].inversearc->adjvex]);
							active_grad = reduce_sum(active_grad, 1);
							active_grad.reShape(graph->vertex[id].shape);
							id = graph->vertex[id].inversearc->adjvex;
						}
						else if (kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == ReShape || kind == Dropout)
						{
							active_grad.reShape(graph->vertex[id].shape);
							active_grad.fill_clear(1);
						}
						else if (kind == Constant)
						{
							return;
						}
						if (isWX)
						{
							weight.flip2d();
							delta = hadamard(matmul(weight, delta), active_grad);
						}
						else
						{
							delta.flip2d();
							//std::cout << id;
							//std::cout << delta;
							delta = matmul(weight, delta);
							delta.flip2d(); 
							delta = hadamard(delta, active_grad);
							
						}
						//gradient[0].flip2d();
						//std::cout << matmul(delta, gradient[0]);
						//return;
					}
					
				}
			}
		}

		void ApplyGrad(int index, float rate)
		{
			VertexKind kind = graph->vertex[index].vkind;
			if (kind == Variable)
			{
				if (graph->vertex[index].shape == gradient[index].shape)
				{
					/*
					if (checknan(gradient[index]))
					{
						return;
					}
					else
					{
						gradient[index] = filternan(gradient[index]);
					}*/
					gradient[index] = filternan(gradient[index]);
					Tensor t = sub(graph->vertex[index], matmul(gradient[index], rate));
					graph->vertex[index].tensor = t.tensor;
				}
				else
				{
					std::cout << index << " gradient error" << std::endl;
				}
			}
			else if (kind == Convolution)
			{
				ApplyGrad(graph->vertex[index].inversearc->adjvex, rate);
				ApplyGrad(graph->vertex[index].inversearc->next->adjvex, rate);
			}
			else if (kind == MaxPooling || kind == AveragePooling || kind == Relu || kind == Softmax || kind == Dropout || kind == ReShape || kind == Reduce_sum)
			{
				ApplyGrad(graph->vertex[index].inversearc->adjvex, rate);
			}
			else if (kind == Addition || kind == Subtraction || kind == Multiplication || kind == Division || kind == Hadamard)
			{
				ApplyGrad(graph->vertex[index].inversearc->adjvex, rate);
				ApplyGrad(graph->vertex[index].inversearc->next->adjvex, rate);
			}
			else if (kind == Exp || kind == Logarithm || kind == Neg || kind == Abs || kind == Square)
			{
				ApplyGrad(graph->vertex[index].inversearc->adjvex, rate);
			}
		}

		bool isEnd(Tensor last,Tensor now)
		{
			int num, num_t;
			double t = 0.0;
			num = now.shape.width*now.shape.height*now.shape.channels*now.shape.variable;
			num_t = 0;
			Tensor diff = abs(sub(last, now));
			auto f = [&](int w, int h, int c, int v) {
				t = abs(last.get(w, h, c, v) - now.get(w, h, c, v));
				if (t < 0.2)
				{
					num_t++;
				}
			};
			traverse(now.shape, f);
			float rate = (double)num_t / num;
			if (rate > 0.2)
			{
				return true;
			}
			return false;
		}

		int getBatch(int index)
		{
			std::vector<int> queue;
			int id = 0;
			queue.push_back(index);
			int batchs = 0;
			while (queue.size() > 0)
			{
				id = queue[0];
				queue.erase(queue.begin());
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == Constant)
				{
					if (graph->vertex[id].batch > batchs)
					{
						batchs = graph->vertex[id].batch;
					}
				}
				ArcNode *p = graph->vertex[id].inversearc;
				while (p != NULL)
				{
					queue.push_back(p->adjvex);
					p = p->next;
				}
			}
			return batchs;
		}

		void nextBatch(int index)
		{
			std::vector<int> queue;
			int id = 0;
			queue.push_back(index);
			int batchs = 0;
			while (queue.size() > 0)
			{
				id = queue[0];
				queue.erase(queue.begin());
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == Constant)
				{
					graph->vertex[id].nextBatch();
				}
				ArcNode *p = graph->vertex[id].inversearc;
				while (p != NULL)
				{
					queue.push_back(p->adjvex);
					p = p->next;
				}
			}
		}

	private:

		int getNextLevel(int index)
		{
			std::vector<int> queue;
			int id = 0;
			queue.push_back(index);
			while (queue.size() > 0)
			{
				id = queue[0];
				queue.erase(queue.begin());
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == ReShape)
				{
					kind = graph->vertex[graph->vertex[id].inversearc->adjvex].vkind;
				}
				if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout)
				{
					return id;
				}
				ArcNode *p = graph->vertex[id].inversearc;
				while (p != NULL)
				{
					queue.push_back(p->adjvex);
					p = p->next;
				}
			}
			if (graph->vertex[id].vkind == Constant)
			{
				return id;
			}
			return -1;
		}

		int getWeight(int index)
		{
			std::vector<int> queue;
			int id = 0;
			queue.push_back(index);
			while (queue.size() > 0)
			{
				id = queue[0];
				queue.erase(queue.begin());
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout)
				{
					continue;
				}
				if (kind == Multiplication)
				{
					int t = getInput(id, input_1);
					kind = graph->vertex[t].vkind;
					if (kind == Variable)
					{
						return graph->vertex[t].nodeid;
					}
					else if (kind == ReShape)
					{
						if (graph->vertex[graph->vertex[t].inversearc->adjvex].vkind == Variable)
						{
							return graph->vertex[t].nodeid;
						}
					}
					t = getInput(id, input_2);
					kind = graph->vertex[t].vkind;
					if (kind == Variable)
					{
						return graph->vertex[t].nodeid;
					}
					else if (kind == ReShape)
					{
						if (graph->vertex[graph->vertex[t].inversearc->adjvex].vkind == Variable)
						{
							return graph->vertex[t].nodeid;
						}
					}
				}
				ArcNode *p = graph->vertex[id].inversearc;
				while (p != NULL)
				{
					queue.push_back(p->adjvex);
					p = p->next;
				}
			}
			return -1;
		}

		int getBias(int index)
		{
			std::vector<int> queue;
			int id = 0;
			queue.push_back(index);
			while (queue.size() > 0)
			{
				id = queue[0];
				queue.erase(queue.begin());
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout)
				{
					continue;
				}
				if (kind == Addition || kind == Subtraction || kind == Neg)
				{
					kind = graph->vertex[getInput(id, input_1)].vkind;
					if (kind == Variable || kind == ReShape)
					{
						return graph->vertex[getInput(id, input_1)].nodeid;
					}
					kind = graph->vertex[getInput(id, input_2)].vkind;
					if (kind == Variable || kind == ReShape)
					{
						return graph->vertex[getInput(id, input_2)].nodeid;
					}
				}
				ArcNode *p = graph->vertex[id].inversearc;
				while (p != NULL)
				{
					queue.push_back(p->adjvex);
					p = p->next;
				}
			}
			return -1;
		}

		int getInput(int index)
		{
			std::vector<int> queue;
			int id = 0;
			queue.push_back(index);
			while (queue.size() > 0)
			{
				id = queue[0];
				queue.erase(queue.begin());
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == Multiplication)
				{
					int t = getInput(id, input_1);
					kind = graph->vertex[t].vkind;
					if (kind == ReShape)
					{
						kind = graph->vertex[graph->vertex[t].inversearc->adjvex].vkind;
						if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout || kind == Constant)
						{
							return t;
						}
					}
					else if (kind == Constant)
					{
						return t;
					}
					t = getInput(id, input_2);
					kind = graph->vertex[t].vkind;
					if (kind == ReShape)
					{
						kind = graph->vertex[graph->vertex[t].inversearc->adjvex].vkind;
						if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout || kind == Constant)
						{
							return t;
						}
					}
					else if (kind == Constant)
					{
						return t;
					}
				}
				if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout || kind == Constant)
				{
					//continue;
					return id;
				}
				ArcNode *p = graph->vertex[id].inversearc;
				while (p != NULL)
				{
					queue.push_back(p->adjvex);
					p = p->next;
				}
			}
			return -1;
		}

		bool checknan(Tensor tensor)
		{
			int nan_num = 0;
			int num = tensor.getShape().width*tensor.getShape().height*tensor.getShape().channels*tensor.getShape().variable;
			auto f = [&](int w, int h, int c, int v) {
				if (isnan(tensor.get(w,h,c,v)))
				{
					nan_num++;
				}
			};
			traverse(tensor.shape, f);
			float rate = (double)nan_num / num;
			if (rate > 0.8)
			{
				return true;
			}
			return false;
		}

		Tensor filternan(Tensor tensor)
		{
			auto f = [&](int w, int h, int c, int v) {
				if (isnan(tensor.get(w, h, c, v)))
				{
					tensor.set(w, h, c, v, 0.0);
				}
			};
			traverse(tensor.shape, f);
			return tensor;
		}

	private:

		int getInput(int index,InfoType type)
		{
			int id = -1;
			if (graph->vertex[index].inversearc->info == type)
			{
				id = graph->vertex[index].inversearc->adjvex;
			}
			else if (graph->vertex[index].inversearc->next->info == type)
			{
				id = graph->vertex[index].inversearc->next->adjvex;
			}
			return id;
		}

		bool judgeWX(int index)
		{
			std::vector<int> queue;
			int id = 0;
			queue.push_back(index);
			while (queue.size() > 0)
			{
				id = queue[0];
				queue.erase(queue.begin());
				VertexKind kind = graph->vertex[id].vkind;
				if (kind == Softmax || kind == Relu || kind == MaxPooling || kind == AveragePooling || kind == Convolution || kind == Dropout)
				{
					continue;
				}
				if (kind == Multiplication)
				{
					kind = graph->vertex[getInput(id, input_1)].vkind;
					if (kind == Variable)
					{
						return true;
					}
					else if (kind == ReShape)
					{
						kind = graph->vertex[graph->vertex[getInput(id, input_1)].inversearc->adjvex].vkind;
						if (kind == Variable)
						{
							return true;
						}
					}
				}
				ArcNode *p = graph->vertex[id].inversearc;
				while (p != NULL)
				{
					queue.push_back(p->adjvex);
					p = p->next;
				}
			}
			return false;
		}

	private:
		Graph *graph = (Graph *)((char *)this - offsetof(Graph, train));
		std::vector<Tensor> gradient;

	}train;

private:

	Tensor& ForwardPropagation(int index)
	{
		VertexKind Kind = vertex[index].vkind;
		if (vertex[index].inversearc == nullptr && (Kind == Variable || Kind == Constant))
		{
			return vertex[index];
		}
		if (index < 0)
		{
		}
		else if (Kind == Variable)
		{
			return vertex[index];
		}
		else if (Kind == Constant)
		{
			return vertex[index];
		}
		else if (Kind == Addition)
		{
			int t1 = -1, t2 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_2)
				{
					t2 = p->adjvex;
				}
				else if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = add(ForwardPropagation(t1), ForwardPropagation(t2));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Subtraction)
		{
			int t1 = -1, t2 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_2)
				{
					t2 = p->adjvex;
				}
				else if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = sub(ForwardPropagation(t1), ForwardPropagation(t2));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Multiplication)
		{
			int t1 = -1, t2 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_2)
				{
					t2 = p->adjvex;
				}
				else if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = matmul(ForwardPropagation(t1), ForwardPropagation(t2));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Division)
		{
		}
		else if (Kind == Neg)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = neg(ForwardPropagation(t1));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Abs)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = abs(ForwardPropagation(t1));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Exp)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = exp(ForwardPropagation(t1));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Logarithm)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = log(ForwardPropagation(t1));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Hadamard)
		{
			int t1 = -1, t2 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_2)
				{
					t2 = p->adjvex;
				}
				else if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = hadamard(ForwardPropagation(t1), ForwardPropagation(t2));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Reduce_sum)
		{
			int t1 = -1, axis = 0;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			axis = vertex[index].par->axis;
			Tensor t = reduce_sum(ForwardPropagation(t1), axis);
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == ReShape)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Shape shape;
			shape = vertex[index].par->newshape;
			Tensor t = ForwardPropagation(t1);
			if (!istrain)
			{
				shape.variable = vertex[t1].getShape().variable;
			}
			t = reshape(t, shape);
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Convolution)
		{
			int t1 = -1, t2 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_2)
				{
					t2 = p->adjvex;
				}
				else if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Shape strides = vertex[index].par->strides;
			padding pad = vertex[index].par->pad;
			Tensor t = conv2d(ForwardPropagation(t1), ForwardPropagation(t2), strides, pad);
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == MaxPooling)
		{
		    int t1 = -1;
		    ArcNode *p = vertex[index].inversearc;
		    while (p != nullptr)
		    {
			    if (p->info == input_1)
			    {
				    t1 = p->adjvex;
			    }
			    p = p->next;
		    }
			Shape ksize = vertex[index].par->ksize;
		    Shape strides = vertex[index].par->strides;
		    padding pad = vertex[index].par->pad;
			Tensor t = max_pool(ForwardPropagation(t1), ksize, strides, pad);
		    vertex[index].tensor = t.tensor;
		    vertex[index].shape = t.shape;
		    return vertex[index];
        }
		else if (Kind == AveragePooling)
		{
		    int t1 = -1;
		    ArcNode *p = vertex[index].inversearc;
		    while (p != nullptr)
		    {
			    if (p->info == input_1)
			    {
				t1 = p->adjvex;
		     	}
			    p = p->next;
		    }
		    Shape ksize = vertex[index].par->ksize;
		    Shape strides = vertex[index].par->strides;
		    padding pad = vertex[index].par->pad;
		    Tensor t = average_pool(ForwardPropagation(t1), ksize, strides, pad);
		    vertex[index].tensor = t.tensor;
		    vertex[index].shape = t.shape;
		    return vertex[index];
		}
		else if (Kind == Relu)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = relu(ForwardPropagation(t1));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
		else if (Kind == Dropout)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			float keep_prob = vertex[index].par->keep_prob;
			if (istrain)
			{
				Tensor t = dropout(ForwardPropagation(t1), keep_prob);
				vertex[index].tensor = t.tensor;
				vertex[index].shape = t.shape;
			}
			else
			{
				Tensor t = ForwardPropagation(t1);
				vertex[index].tensor = t.tensor;
				vertex[index].shape = t.shape;
			}
			return vertex[index];
		}
		else if (Kind == Softmax)
		{
			int t1 = -1;
			ArcNode *p = vertex[index].inversearc;
			while (p != nullptr)
			{
				if (p->info == input_1)
				{
					t1 = p->adjvex;
				}
				p = p->next;
			}
			Tensor t = softmax(ForwardPropagation(t1));
			vertex[index].tensor = t.tensor;
			vertex[index].shape = t.shape;
			return vertex[index];
		}
	}

private:
	
	std::vector<Tensor> vertex;
	int vexnum, arcnum;
	bool istrain;

};
