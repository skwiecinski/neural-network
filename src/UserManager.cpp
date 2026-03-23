#include "UserManager.h"
#include <filesystem> 

namespace fs = std::filesystem;

UserManager::UserManager(const std::string& usersFilePath) : usersFilePath(usersFilePath) {
    loadUsers();
}

void UserManager::loadUsers() {
    users.clear();
    std::ifstream file(usersFilePath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string username, password;
            if (std::getline(ss, username, ',') && std::getline(ss, password, ',')) {
                users.emplace_back(username, password);
            }
        }
        file.close();
    }
}

void UserManager::saveUsers() {
    std::ofstream file(usersFilePath);
    if (file.is_open()) {
        for (const auto& user : users) {
            file << user.username << "," << user.password << ",\n";
        }
        file.close();
    }
    else {
        std::cerr << "Error: Could not save user data to " << usersFilePath << "\n";
    }
}

bool UserManager::registerUser(const std::string& username, const std::string& password) {
    if (username.empty() || password.empty()) {
        std::cout << "Username and password cannot be empty.\n";
        return false;
    }
    for (const auto& user : users) {
        if (user.username == username) {
            std::cout << "Username already taken. Please choose another.\n";
            return false;
        }
    }
    users.emplace_back(username, password);
    saveUsers();
    std::string userDir = getUserModelPath(username);
    if (!fs::exists(userDir)) {
        fs::create_directories(userDir);
    }
    std::cout << "User '" << username << "' registered successfully.\n";
    return true;
}

bool UserManager::loginUser(const std::string& username, const std::string& password) {
    if (username.empty() || password.empty()) {
        std::cout << "Username and password cannot be empty.\n";
        return false;
    }
    for (const auto& user : users) {
        if (user.username == username && user.password == password) {
            std::cout << "User '" << username << "' logged in successfully.\n";
            return true;
        }
    }
    std::cout << "Invalid username or password.\n";
    return false;
}

std::string UserManager::getUserModelPath(const std::string& username) const {
    return "models/" + username;
}