#include "image_generator.h"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "../stable-diffusion.cpp/thirdparty/stb_image_write.h"
#include "base64.hpp"
#include "logging.h"

// Global callback data structure
struct CallbackData {
    ImageGenerator* generator;
    std::string task_id;
    TaskStateManager* task_state_manager;
    int total_steps;
};

static CallbackData g_callback_data;
static std::mutex g_callback_mutex;

// Progress callback for stable-diffusion.cpp
static void progress_callback(int step, int steps, float time, void* data) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    if (g_callback_data.task_state_manager && !g_callback_data.task_id.empty()) {
        float progress = steps > 0 ? static_cast<float>(step) / static_cast<float>(steps) : 0.0f;
        g_callback_data.task_state_manager->updateTaskProgress(g_callback_data.task_id, progress);
        LOG_DEBUG("Progress: step %d/%d (%.1f%%), time: %.2fs", step, steps, progress * 100.0f, time);
    }
}

// Preview callback for stable-diffusion.cpp
static void preview_callback(int step, int frame_count, sd_image_t* frames, bool is_noisy) {
    std::lock_guard<std::mutex> lock(g_callback_mutex);
    if (g_callback_data.task_state_manager && !g_callback_data.task_id.empty() && frames && frame_count > 0) {
        // Encode first frame as base64 for live preview
        size_t img_size = frames[0].width * frames[0].height * frames[0].channel;
        std::string preview_base64 = base64_encode(frames[0].data, img_size);

        float progress = g_callback_data.total_steps > 0
                             ? static_cast<float>(step) / static_cast<float>(g_callback_data.total_steps)
                             : 0.0f;

        g_callback_data.task_state_manager->updateTaskProgress(g_callback_data.task_id, progress, preview_base64);

        LOG_DEBUG("Preview: step %d, frames: %d, noisy: %d", step, frame_count, is_noisy);
    }
}

ImageGenerator::ImageGenerator(std::shared_ptr<TaskStateManager> task_state_manager)
    : sd_ctx_(nullptr), task_state_manager_(task_state_manager), initialized_(false), interrupted_(false) {
    LOG_INFO("ImageGenerator created");
}

ImageGenerator::~ImageGenerator() {
    if (sd_ctx_) {
        LOG_INFO("Freeing SD context");
        free_sd_ctx(sd_ctx_);
        sd_ctx_ = nullptr;
    }
}

bool ImageGenerator::initialize(const std::string& model_path, const std::string& vae_path,
                                const std::string& taesd_path, const std::string& lora_model_dir,
                                const std::string& embeddings_dir, int n_threads, sd_type_t wtype, bool offload_to_cpu,
                                bool vae_on_cpu) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sd_ctx_) {
        LOG_WARNING("SD context already initialized, freeing old context");
        free_sd_ctx(sd_ctx_);
        sd_ctx_ = nullptr;
    }

    if (model_path.empty()) {
        LOG_ERROR("Model path is empty");
        return false;
    }

    LOG_INFO("Initializing SD context with model: %s", model_path.c_str());

    // Initialize context parameters
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);

    params.free_params_immediately = false;

    params.model_path = model_path.c_str();
    params.vae_path = vae_path.empty() ? nullptr : vae_path.c_str();
    params.taesd_path = taesd_path.empty() ? nullptr : taesd_path.c_str();
    params.lora_model_dir = lora_model_dir.empty() ? nullptr : lora_model_dir.c_str();
    params.embedding_dir = embeddings_dir.empty() ? nullptr : embeddings_dir.c_str();
    params.n_threads = n_threads;
    params.wtype = wtype;
    params.offload_params_to_cpu = offload_to_cpu;
    params.keep_vae_on_cpu = vae_on_cpu;
    params.vae_decode_only = false;  // We need encoding for img2img

    // Set RNG type
    params.rng_type = CUDA_RNG;

    // Create SD context
    sd_ctx_ = new_sd_ctx(&params);
    if (!sd_ctx_) {
        LOG_ERROR("Failed to create SD context");
        return false;
    }

    initialized_ = true;
    LOG_INFO("SD context initialized successfully");

    return true;
}

bool ImageGenerator::isInitialized() const { return initialized_ && sd_ctx_ != nullptr; }

void ImageGenerator::interrupt() {
    std::lock_guard<std::mutex> lock(mutex_);
    interrupted_ = true;
    LOG_INFO("Generation interrupted");
}

std::vector<std::string> ImageGenerator::generateTxt2Img(const ImageGenerationParams& params,
                                                         const std::string& task_id) {
    return generateInternal(params, false, task_id);
}

std::vector<std::string> ImageGenerator::generateImg2Img(const ImageGenerationParams& params,
                                                         const std::string& task_id) {
    return generateInternal(params, true, task_id);
}

std::vector<std::string> ImageGenerator::generateInternal(const ImageGenerationParams& params, bool is_img2img,
                                                          const std::string& task_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!isInitialized()) {
        LOG_ERROR("SD context not initialized");
        throw std::runtime_error("SD context not initialized");
    }

    interrupted_ = false;
    current_task_id_ = task_id;

    // Set up callbacks
    {
        std::lock_guard<std::mutex> cb_lock(g_callback_mutex);
        g_callback_data.generator = this;
        g_callback_data.task_id = task_id;
        g_callback_data.task_state_manager = task_state_manager_.get();
        g_callback_data.total_steps = params.steps;
    }

    sd_set_progress_callback(progress_callback, nullptr);
    sd_set_preview_callback(preview_callback, PREVIEW_NONE, 1, true, false);

    LOG_INFO("Generating %s: prompt='%s', size=%dx%d, steps=%d, seed=%lld", is_img2img ? "img2img" : "txt2img",
             params.prompt.c_str(), params.width, params.height, params.steps, params.seed);

    // Initialize generation parameters
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    gen_params.prompt = params.prompt.c_str();
    gen_params.negative_prompt = params.negative_prompt.empty() ? "" : params.negative_prompt.c_str();
    gen_params.clip_skip = params.clip_skip;
    gen_params.width = params.width;
    gen_params.height = params.height;
    gen_params.batch_count = params.batch_count;
    gen_params.seed = params.seed < 0 ? static_cast<int64_t>(time(nullptr)) : params.seed;

    // Sample parameters
    gen_params.sample_params.sample_method = params.sampler;
    gen_params.sample_params.scheduler = params.scheduler;
    gen_params.sample_params.sample_steps = params.steps;
    gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;

    // img2img specific
    sd_image_t init_image = {0, 0, 0, nullptr};
    if (is_img2img && !params.init_image_base64.empty()) {
        init_image = base64ToImage(params.init_image_base64);
        gen_params.init_image = init_image;
        gen_params.strength = params.strength;
    }

    // Generate images
    sd_image_t* result = generate_image(sd_ctx_, &gen_params);

    // Free init image if used
    if (init_image.data) {
        freeImage(init_image);
    }

    if (!result) {
        LOG_ERROR("Image generation failed");

        // Clear callbacks
        {
            std::lock_guard<std::mutex> cb_lock(g_callback_mutex);
            g_callback_data.task_id.clear();
        }

        throw std::runtime_error("Image generation failed");
    }

    // Convert results to base64
    std::vector<std::string> result_images;
    for (int i = 0; i < params.batch_count; i++) {
        std::string img_base64 = imageToBase64(result[i]);
        result_images.push_back(img_base64);

        // Free the result image data
        if (result[i].data) {
            free(result[i].data);
        }
    }

    // Free result array
    free(result);

    // Clear callbacks
    {
        std::lock_guard<std::mutex> cb_lock(g_callback_mutex);
        g_callback_data.task_id.clear();
    }

    LOG_INFO("Generated %zu images successfully", result_images.size());

    return result_images;
}

std::string ImageGenerator::imageToBase64(const sd_image_t& image) {
    if (!image.data || image.width == 0 || image.height == 0) {
        return "";
    }

    // Use stb_image_write to convert to PNG format
    std::vector<unsigned char> png_buffer;

    // Lambda function to capture PNG data
    auto write_callback = [](void* context, void* data, int size) {
        auto* buffer = static_cast<std::vector<unsigned char>*>(context);
        unsigned char* bytes = static_cast<unsigned char*>(data);
        buffer->insert(buffer->end(), bytes, bytes + size);
    };

    // Write PNG to buffer (stride is width * channels for tightly packed data)
    int stride = image.width * image.channel;
    int result = stbi_write_png_to_func(write_callback, &png_buffer, image.width, image.height, image.channel,
                                        image.data, stride);

    if (result == 0 || png_buffer.empty()) {
        LOG_ERROR("Failed to encode image as PNG");
        return "";
    }

    // Encode PNG data as base64
    return base64_encode(png_buffer.data(), png_buffer.size());
}

sd_image_t ImageGenerator::base64ToImage(const std::string& base64_data) {
    sd_image_t image = {0, 0, 0, nullptr};

    if (base64_data.empty()) {
        return image;
    }

    // Decode base64
    std::vector<uint8_t> decoded = base64_decode(base64_data);
    if (decoded.empty()) {
        LOG_ERROR("Failed to decode base64 image data");
        return image;
    }

    // For simplicity, we'll need to parse the image format
    // This is a simplified version - in production, you'd want to use stb_image or similar
    // For now, assume raw RGB data and extract dimensions from somewhere
    // This needs to be improved with proper image format handling

    LOG_WARNING("base64ToImage needs proper image format parsing implementation");

    return image;
}

void ImageGenerator::freeImage(sd_image_t& image) {
    if (image.data) {
        free(image.data);
        image.data = nullptr;
    }
    image.width = 0;
    image.height = 0;
    image.channel = 0;
}
