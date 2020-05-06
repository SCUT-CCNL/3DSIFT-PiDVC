#include "../Include/PriorityQueue.h"


PriorityQueue::PriorityQueue() {}

bool PriorityQueue::getFrontAndRemove(NPOI*&p) {
	return pqueue.try_pop(p);
}

void PriorityQueue::insert(NPOI *p) {
	pqueue.push(p);
}



bool PriorityQueue::isEmpty() {
	return pqueue.empty();
}

int PriorityQueue::size() {
	return (int)pqueue.size();
}



PriorityQueue::~PriorityQueue() {
	pqueue.clear();
}