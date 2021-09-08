#pragma once

#include <string>
#include <vector>

namespace StrProcUtils
{
std::vector<std::string> charItemsToVector(const char* ch);
std::vector<std::string> stringToStringVector(std::string s);
std::string stringVectorToString(std::vector<std::string> svec);
std::string stringVectorToStringWithoutSpaces(std::vector<std::string> svec);
bool isPrefix(std::string str1, std::string str2);
// returns true if string str1 is a prefix of string str2
bool isPrefixStrVec(std::vector<std::string> strVec1, std::vector<std::string> strVec2);
// returns true if string vector strVec1 is a prefix of string
// vector strVec2
std::string getLastWord(std::string str);
// Remove last word contained in string str
bool lastCharIsBlank(const std::string& str);
// Returns true if last char of str is blank
std::string removeLastBlank(std::string str);
// Remove last blank character of str if exists
std::string addBlank(std::string str);
// Add blank character at the end of str
std::vector<float> strVecToFloatVec(std::vector<std::string> strVec);
// Convert string vector into a float vector
std::vector<std::string> split(const std::string& s, char delim);
} // namespace StrProcUtils
