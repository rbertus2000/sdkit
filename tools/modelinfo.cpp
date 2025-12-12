#include <iostream>
#include <string>
#include <vector>

#include "../include/model_detection.h"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path>\n";
    std::cout << "\n";
    std::cout << "Description:\n";
    std::cout << "  Inspects a GGUF or safetensors model file and prints information about it.\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  model_path    Path to the model file (.gguf or .safetensors)\n";
    std::cout << "\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " path/to/model.safetensors\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Error: Invalid number of arguments.\n\n";
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path = argv[1];

    // Detect model format
    std::string format = detectModelFormat(model_path);

    if (format == "unknown") {
        std::cerr << "Error: Unknown or unsupported file format.\n";
        std::cerr << "Supported formats: GGUF (.gguf), SafeTensors (.safetensors)\n";
        return 1;
    }

    // Extract tensor keys
    std::vector<std::string> tensor_keys = extractTensorKeys(model_path, format);
    if (tensor_keys.empty()) {
        std::cerr << "Error: Failed to read model file or no tensors found.\n";
        return 1;
    }

    // Detect model type
    std::string model_type = inferModelTypeFromTensorKeys(tensor_keys);
    std::cout << "Model type: " << model_type << "\n";

    std::cout << "Number of tensors: " << tensor_keys.size() << "\n";

    // // Print first few tensor names as samples
    // std::cout << "\nSample tensor names:\n";
    // size_t sample_count = std::min(static_cast<size_t>(10), tensor_keys.size());
    // for (size_t i = 0; i < sample_count; i++) {
    //     std::cout << "  " << (i + 1) << ". " << tensor_keys[i] << "\n";
    // }

    // if (tensor_keys.size() > sample_count) {
    //     std::cout << "  ... and " << (tensor_keys.size() - sample_count) << " more tensors\n";
    // }

    // std::cout << "\n=== Analysis Complete ===\n";
    return 0;
}
