#ifndef USER_MANAGER_H
#define USER_MANAGER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream> 

#include "User.h"

class UserManager {
public:
    UserManager(const std::string& usersFilePath = "models/users.dat"); 

    bool registerUser(const std::string& username, const std::string& password);
    bool loginUser(const std::string& username, const std::string& password);

    std::string getUserModelPath(const std::string& username) const;

private:
    std::string usersFilePath;
    std::vector<User> users;

    void loadUsers();
    void saveUsers();
};

#endif