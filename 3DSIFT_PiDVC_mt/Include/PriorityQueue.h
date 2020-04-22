#pragma once
#ifndef PIDVC_PRIORITY_QUEUE
#define PIDVC_PRIORITY_QUEUE

#include"POI.h"
#include<concurrent_priority_queue.h> //perhaps windows only

using concurrency::concurrent_priority_queue;

class PriorityQueue
{
	concurrent_priority_queue<NPOI*, NPOI_p_Compare_Neighbour_ZNCC> pqueue;
public:
	PriorityQueue();
	~PriorityQueue();


	bool getFrontAndRemove(NPOI*&);
	void insert(NPOI*);
	bool isEmpty();
	int size();

};


#endif // ! PDDVC_PRIORITY_QUEUE