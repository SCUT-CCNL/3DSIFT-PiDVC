#ifndef __PIDVC_CONF_H__
#define __PIDVC_CONF_H__

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

template <class T>
T getValue(std::map<std::string, T> &map, const std::string key) {
	T temp;
	auto it = map.find(key);
	if (it == map.end()) {
		std::cerr << "Error : fail to find key\"" << key << "\" in parameters map. " << std::endl;
		throw "fail to access key";
	}
	else {
		temp = map[key];
	}
	return temp;
}

#endif // ! __UTIL_H__