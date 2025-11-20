#include <csignal>
#include <iostream>
#include <memory>

#include "logging.h"
#include "server.h"

std::unique_ptr<Server> g_server;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

int main(int argc, char* argv[]) {
    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Default port
    int port = 8188;

    // Parse command line arguments
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (const std::exception& e) {
            std::cerr << "Invalid port number: " << argv[1] << std::endl;
            std::cerr << "Usage: " << argv[0] << " [port]" << std::endl;
            return 1;
        }
    }

    std::cout << "==================================" << std::endl;
    std::cout << "  Stable Diffusion API Server" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Port: " << port << std::endl;
    std::cout << std::endl;

    try {
        // Create and start the server
        g_server = std::make_unique<Server>(port);
        g_server->run();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server stopped." << std::endl;
    return 0;
}
