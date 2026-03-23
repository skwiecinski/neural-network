#ifndef USER_H
#define USER_H

#include <string>

struct User {
    std::string username;
    std::string password; 

    User(const std::string& user, const std::string& pass) : username(user), password(pass) {}
};

#endif 