#include "model_manager.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

#include "logging.h"

namespace fs = std::filesystem;

// Define valid model file extensions
const std::vector<std::string> ModelManager::valid_extensions_ = {".sft", ".safetensors", ".pth",
                                                                  ".pt",  ".ckpt",        ".gguf"};

ModelManager::ModelManager() { LOG_INFO("ModelManager initialized"); }

ModelManager::~ModelManager() {}

void ModelManager::setCheckpointDir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::CHECKPOINT] = dir;
    LOG_INFO("Checkpoint directory set to: %s", dir.c_str());
}

void ModelManager::setVaeDir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::VAE] = dir;
    LOG_INFO("VAE directory set to: %s", dir.c_str());
}

void ModelManager::setHypernetworkDir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::HYPERNETWORK] = dir;
    LOG_INFO("Hypernetwork directory set to: %s", dir.c_str());
}

void ModelManager::setGfpganModelsPath(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::GFPGAN] = dir;
    LOG_INFO("GFPGAN models path set to: %s", dir.c_str());
}

void ModelManager::setRealesrganModelsPath(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::REALESRGAN] = dir;
    LOG_INFO("RealESRGAN models path set to: %s", dir.c_str());
}

void ModelManager::setLoraDir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::LORA] = dir;
    LOG_INFO("LoRA directory set to: %s", dir.c_str());
}

void ModelManager::setCodeformerModelsPath(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::CODEFORMER] = dir;
    LOG_INFO("Codeformer models path set to: %s", dir.c_str());
}

void ModelManager::setEmbeddingsDir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::EMBEDDINGS] = dir;
    LOG_INFO("Embeddings directory set to: %s", dir.c_str());
}

void ModelManager::setControlnetDir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::CONTROLNET] = dir;
    LOG_INFO("ControlNet directory set to: %s", dir.c_str());
}

void ModelManager::setTextEncoderDir(const std::string& dir) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_directories_[ModelType::TEXT_ENCODER] = dir;
    LOG_INFO("Text Encoder directory set to: %s", dir.c_str());
}

std::string ModelManager::getRealesrganModelsPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = model_directories_.find(ModelType::REALESRGAN);
    return it != model_directories_.end() ? it->second : "";
}

std::string ModelManager::getLoraDir() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = model_directories_.find(ModelType::LORA);
    return it != model_directories_.end() ? it->second : "";
}

bool ModelManager::isValidModelFile(const std::string& filename) const {
    std::string lower_filename = filename;
    std::transform(lower_filename.begin(), lower_filename.end(), lower_filename.begin(), ::tolower);

    for (const auto& ext : valid_extensions_) {
        if (lower_filename.size() >= ext.size() &&
            lower_filename.compare(lower_filename.size() - ext.size(), ext.size(), ext) == 0) {
            return true;
        }
    }
    return false;
}

void ModelManager::scanDirectoryInternal(const std::string& directory, ModelType type) {
    if (directory.empty()) {
        return;
    }

    if (!fs::exists(directory)) {
        LOG_WARNING("Directory does not exist: %s", directory.c_str());
        return;
    }

    if (!fs::is_directory(directory)) {
        LOG_WARNING("Path is not a directory: %s", directory.c_str());
        return;
    }

    std::vector<ModelInfo> found_models;

    try {
        for (const auto& entry : fs::recursive_directory_iterator(directory)) {
            if (!entry.is_regular_file()) {
                continue;
            }

            std::string filename = entry.path().filename().string();
            if (!isValidModelFile(filename)) {
                continue;
            }

            std::string full_path = entry.path().string();
            size_t file_size = fs::file_size(entry.path());

            // Determine the lookup name based on model type
            std::string lookup_name;
            if (type == ModelType::CHECKPOINT) {
                // For stable-diffusion models: use relative path from directory with extension
                fs::path relative = fs::relative(entry.path(), directory);
                lookup_name = relative.string();
            } else if (type == ModelType::CONTROLNET || type == ModelType::EMBEDDINGS || type == ModelType::LORA) {
                // For controlnet, embeddings, lora: use filename without extension
                lookup_name = entry.path().stem().string();
            } else {
                // For other model types: use filename as-is
                lookup_name = filename;
            }

            ModelInfo info(lookup_name, full_path, type, file_size);
            found_models.push_back(info);

            LOG_DEBUG("Found %s model: %s (%zu bytes)", getModelTypeString(type).c_str(), lookup_name.c_str(),
                      file_size);
        }
    } catch (const fs::filesystem_error& e) {
        LOG_ERROR("Error scanning directory %s: %s", directory.c_str(), e.what());
        return;
    }

    // Update the models_ map (already locked by caller)
    models_[type] = found_models;
    LOG_INFO("Scanned %s directory: found %zu models", getModelTypeString(type).c_str(), found_models.size());
}

void ModelManager::scanDirectory(ModelType type) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = model_directories_.find(type);
    if (it == model_directories_.end() || it->second.empty()) {
        LOG_WARNING("No directory set for model type: %s", getModelTypeString(type).c_str());
        return;
    }

    scanDirectoryInternal(it->second, type);
}

void ModelManager::scanAllDirectories() {
    std::lock_guard<std::mutex> lock(mutex_);

    LOG_INFO("Scanning all model directories...");

    for (const auto& [type, directory] : model_directories_) {
        if (!directory.empty()) {
            scanDirectoryInternal(directory, type);
        }
    }

    LOG_INFO("Finished scanning all model directories");
}

void ModelManager::refresh() {
    LOG_INFO("Refreshing all models...");
    scanAllDirectories();
}

void ModelManager::refreshCheckpoints() {
    LOG_INFO("Refreshing checkpoints...");
    scanDirectory(ModelType::CHECKPOINT);
}

void ModelManager::refreshVaeAndTextEncoders() {
    LOG_INFO("Refreshing VAEs and text encoders...");
    scanDirectory(ModelType::VAE);
    scanDirectory(ModelType::TEXT_ENCODER);
}

std::vector<ModelInfo> ModelManager::getModelsByType(ModelType type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(type);
    if (it != models_.end()) {
        return it->second;
    }
    return std::vector<ModelInfo>();
}

std::vector<std::string> ModelManager::getModelNamesByType(ModelType type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> names;
    auto it = models_.find(type);
    if (it != models_.end()) {
        for (const auto& model : it->second) {
            names.push_back(model.filename);
        }
    }
    return names;
}

ModelInfo ModelManager::getModelByName(const std::string& name, ModelType type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(type);
    if (it != models_.end()) {
        for (const auto& model : it->second) {
            if (model.filename == name) {
                return model;
            }
        }
    }
    return ModelInfo();
}

bool ModelManager::hasModel(const std::string& name, ModelType type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = models_.find(type);
    if (it != models_.end()) {
        for (const auto& model : it->second) {
            if (model.filename == name) {
                return true;
            }
        }
    }
    return false;
}

std::map<std::string, std::vector<std::string>> ModelManager::getAllModelsGrouped() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, std::vector<std::string>> grouped;

    for (const auto& [type, model_list] : models_) {
        std::string type_str = getModelTypeString(type);
        std::vector<std::string> names;
        for (const auto& model : model_list) {
            names.push_back(model.filename);
        }
        grouped[type_str] = names;
    }

    return grouped;
}

std::string ModelManager::getModelTypeString(ModelType type) const {
    switch (type) {
        case ModelType::CHECKPOINT:
            return "checkpoint";
        case ModelType::VAE:
            return "vae";
        case ModelType::HYPERNETWORK:
            return "hypernetwork";
        case ModelType::GFPGAN:
            return "gfpgan";
        case ModelType::REALESRGAN:
            return "realesrgan";
        case ModelType::LORA:
            return "lora";
        case ModelType::CODEFORMER:
            return "codeformer";
        case ModelType::EMBEDDINGS:
            return "embeddings";
        case ModelType::CONTROLNET:
            return "controlnet";
        case ModelType::TEXT_ENCODER:
            return "text_encoder";
        default:
            return "unknown";
    }
}
