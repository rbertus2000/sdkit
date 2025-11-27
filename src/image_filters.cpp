#include "image_filters.h"

#include <filesystem>

#include "image_utils.h"
#include "logging.h"

namespace fs = std::filesystem;

ImageFilters::ImageFilters(std::shared_ptr<ModelManager> model_manager)
    : model_manager_(model_manager), upscaler_ctx_(nullptr) {
    LOG_DEBUG("ImageFilters created");
}

ImageFilters::~ImageFilters() {
    if (upscaler_ctx_) {
        LOG_INFO("Freeing upscaler context");
        free_upscaler_ctx(upscaler_ctx_);
        upscaler_ctx_ = nullptr;
    }
}

std::vector<std::string> ImageFilters::upscaleBatch(const std::vector<std::string>& base64_images,
                                                    const std::string& upscaler_name, int upscale_factor) {
    std::vector<std::string> result_images;
    if (!ensureUpscalerLoaded(upscaler_name)) {
        LOG_ERROR("Upscaler not available. Cannot upscale images.");
        return result_images;
    }

    for (size_t i = 0; i < base64_images.size(); i++) {
        // Convert base64 to sd_image_t
        sd_image_t input_image = base64ToImage(base64_images[i]);
        if (!input_image.data) {
            LOG_ERROR("Failed to decode image %zu", i);
            result_images.push_back("");
            continue;
        }

        // Upscale the image
        sd_image_t upscaled_image = upscaleImage(input_image, upscale_factor);

        // Free input image
        freeImage(input_image);

        if (!upscaled_image.data) {
            LOG_ERROR("Failed to upscale image %zu", i);
            result_images.push_back("");
            continue;
        }

        // Convert upscaled image back to base64
        std::string upscaled_base64 = imageToBase64(upscaled_image);
        result_images.push_back(upscaled_base64);

        // Free upscaled image
        freeImage(upscaled_image);

        LOG_INFO("Upscaled image %zu: %dx%d -> %dx%d", i, input_image.width, input_image.height, upscaled_image.width,
                 upscaled_image.height);
    }

    return result_images;
}

sd_image_t ImageFilters::upscaleImage(const sd_image_t& input_image, int upscale_factor) {
    sd_image_t upscaled_image = {0, 0, 0, nullptr};

    if (!input_image.data) {
        LOG_ERROR("Cannot upscale invalid image");
        return upscaled_image;
    }

    if (!upscaler_ctx_) {
        LOG_ERROR("Upscaler not loaded");
        return upscaled_image;
    }

    LOG_DEBUG("Upscaling image from %dx%d, factor %d", input_image.width, input_image.height, upscale_factor);
    upscaled_image = upscale(upscaler_ctx_, input_image, upscale_factor);

    if (upscaled_image.data) {
        LOG_DEBUG("Upscaled to %dx%d", upscaled_image.width, upscaled_image.height);
    }

    return upscaled_image;
}

bool ImageFilters::ensureUpscalerLoaded(const std::string& upscaler_name) {
    // Upscaler name is required
    if (upscaler_name.empty()) {
        LOG_ERROR("No upscaler model name specified");
        return false;
    }

    // If upscaler is already loaded with the same name, we're good
    if (upscaler_ctx_ && current_upscaler_path_ == upscaler_name) {
        return true;
    }

    // Free old upscaler if exists
    if (upscaler_ctx_) {
        LOG_INFO("Freeing old upscaler context");
        free_upscaler_ctx(upscaler_ctx_);
        upscaler_ctx_ = nullptr;
        current_upscaler_path_.clear();
    }

    // Get the realesrgan models directory
    std::string realesrgan_dir = model_manager_->getRealesrganModelsPath();
    if (realesrgan_dir.empty()) {
        throw std::exception("RealESRGAN models directory not set. Use --realesrgan-models-path argument.");
    }

    // Construct full path to the model
    fs::path model_path = fs::path(realesrgan_dir) / (upscaler_name + ".pth");
    std::string upscaler_path = model_path.string();

    LOG_INFO("Loading upscaler from: %s", upscaler_path.c_str());

    upscaler_ctx_ = new_upscaler_ctx(upscaler_path.c_str(), false, false, -1);

    if (!upscaler_ctx_) {
        LOG_ERROR("Failed to load upscaler from: %s", upscaler_path.c_str());
        return false;
    }

    current_upscaler_path_ = upscaler_name;
    LOG_INFO("Upscaler loaded successfully");

    return true;
}