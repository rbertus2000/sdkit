#include "image_generator.h"

#include <cstring>
#include <stdexcept>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "../stable-diffusion.cpp/thirdparty/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stable-diffusion.cpp/thirdparty/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stable-diffusion.cpp/thirdparty/stb_image_resize.h"
#include "base64.hpp"
#include "crow.h"
#include "logging.h"
#include "model_detection.h"

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
    LOG_DEBUG("Preview callback: step %d, frame_count %d, is_noisy %d", step, frame_count, is_noisy);
    if (g_callback_data.task_state_manager && !g_callback_data.task_id.empty() && frames && frame_count > 0) {
        // Encode first frame to JPEG, then base64 for live preview
        std::vector<unsigned char> jpg_buffer;
        auto write_callback = [](void* context, void* data, int size) {
            auto* buffer = static_cast<std::vector<unsigned char>*>(context);
            unsigned char* bytes = static_cast<unsigned char*>(data);
            buffer->insert(buffer->end(), bytes, bytes + size);
        };

        // Use lower quality for faster preview encoding
        int quality = 75;
        int result = stbi_write_jpg_to_func(write_callback, &jpg_buffer, frames[0].width, frames[0].height,
                                            frames[0].channel, frames[0].data, quality);

        LOG_DEBUG("JPEG encoding result: %d, jpg_buffer size: %zu", result, jpg_buffer.size());
        if (result != 0 && !jpg_buffer.empty()) {
            std::string preview_base64 = base64_encode(jpg_buffer.data(), jpg_buffer.size());

            float progress = g_callback_data.total_steps > 0
                                 ? static_cast<float>(step) / static_cast<float>(g_callback_data.total_steps)
                                 : 0.0f;

            g_callback_data.task_state_manager->updateTaskProgress(g_callback_data.task_id, progress, preview_base64);

            LOG_DEBUG("Preview: step %d, frames: %d, noisy: %d, jpg_size: %zu bytes", step, frame_count, is_noisy,
                      jpg_buffer.size());
        } else {
            LOG_ERROR("Failed to encode preview image as JPEG");
        }
    }
}

ImageGenerator::ImageGenerator(std::shared_ptr<TaskStateManager> task_state_manager,
                               std::shared_ptr<OptionsManager> options_manager,
                               std::shared_ptr<ModelManager> model_manager)
    : sd_ctx_(nullptr),
      task_state_manager_(task_state_manager),
      options_manager_(options_manager),
      model_manager_(model_manager),
      initialized_(false),
      interrupted_(false) {
    LOG_INFO("ImageGenerator created");
}

ImageGenerator::~ImageGenerator() {
    if (sd_ctx_) {
        LOG_INFO("Freeing SD context");
        free_sd_ctx(sd_ctx_);
        sd_ctx_ = nullptr;
    }
}

bool ImageGenerator::isInitialized() const { return initialized_ && sd_ctx_ != nullptr; }

std::string ImageGenerator::getCurrentModelPath() const { return current_model_path_; }

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
    // Ensure the correct model is loaded based on current options (before taking lock)
    if (!ensureModelLoaded()) {
        LOG_ERROR("Failed to ensure model is loaded");
        throw std::runtime_error("Failed to load model from options");
    }

    std::lock_guard<std::mutex> lock(mutex_);

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

    // Set preview callback only if live previews are enabled
    auto options = options_manager_->getOptions();
    std::string options_json = options.dump();
    auto parsed_options = crow::json::load(options_json);
    bool live_previews_enabled = true;  // default to true
    if (parsed_options && parsed_options.has("live_previews_enable")) {
        live_previews_enabled = parsed_options["live_previews_enable"].b();
    }

    if (live_previews_enabled) {
        sd_set_preview_callback(preview_callback, PREVIEW_PROJ, 3, true, false);
    } else {
        // Clear any existing preview callback
        sd_set_preview_callback(nullptr, PREVIEW_PROJ, 3, true, false);
    }

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
    if (is_img2img && !params.init_image_base64.empty()) {
        sd_image_t init_image = createInitImage(params);
        sd_image_t mask_image = createMaskImage(params);

        gen_params.init_image = init_image;
        gen_params.mask_image = mask_image;
        gen_params.strength = params.strength;
    }

    // Generate images
    sd_image_t* result = generate_image(sd_ctx_, &gen_params);

    // Free init image if used
    if (gen_params.init_image.data) {
        freeImage(gen_params.init_image);
    }
    if (gen_params.mask_image.data) {
        freeImage(gen_params.mask_image);
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

sd_image_t ImageGenerator::base64ToImage(const std::string& base64_data, int desired_channels) {
    sd_image_t image = {0, 0, 0, nullptr};

    if (base64_data.empty()) {
        return image;
    }

    // Strip data URI prefix if present (e.g., "data:image/png;base64,")
    std::string base64_clean = base64_data;
    size_t comma_pos = base64_data.find(',');
    if (comma_pos != std::string::npos) {
        // Check if this looks like a data URI
        if (base64_data.substr(0, 5) == "data:") {
            base64_clean = base64_data.substr(comma_pos + 1);
            LOG_DEBUG("Stripped data URI prefix, base64 length: %zu", base64_clean.length());
        }
    }

    // Decode base64
    std::vector<uint8_t> decoded = base64_decode(base64_clean);
    if (decoded.empty()) {
        LOG_ERROR("Failed to decode base64 image data");
        return image;
    }

    // Load image from memory using stb_image
    int width, height, channels;
    unsigned char* data = stbi_load_from_memory(decoded.data(), static_cast<int>(decoded.size()), &width, &height,
                                                &channels, desired_channels);
    if (!data) {
        LOG_ERROR("Failed to load image from decoded data: %s", stbi_failure_reason());
        return image;
    }

    // Set image properties
    image.width = width;
    image.height = height;
    image.channel = desired_channels;
    image.data = data;

    LOG_INFO("Loaded image: %dx%d, channels: %d", image.width, image.height, image.channel);

    return image;
}

bool ImageGenerator::resizeImage(sd_image_t& image, int target_width, int target_height) {
    if (!image.data || image.width == 0 || image.height == 0) {
        return false;
    }

    if (target_width <= 0 || target_height <= 0) {
        return false;
    }

    // Clamp to multiples of 8 like in Python code
    int final_width = target_width - (target_width % 8);
    int final_height = target_height - (target_height % 8);

    // If already the right size, do nothing
    if (final_width == image.width && final_height == image.height) {
        LOG_DEBUG("Image already correct size %dx%d", image.width, image.height);
        return true;
    }

    LOG_INFO("Resizing image from %dx%d to %dx%d", image.width, image.height, final_width, final_height);

    // Allocate memory for resized image
    unsigned char* resized_data = static_cast<unsigned char*>(malloc(final_width * final_height * image.channel));
    if (!resized_data) {
        LOG_ERROR("Failed to allocate memory for resized image");
        return false;
    }

    // Use stb_image_resize for high-quality resampling
    int result = stbir_resize_uint8(image.data, image.width, image.height, 0, resized_data, final_width, final_height,
                                    0, image.channel);

    if (result == 0) {
        LOG_ERROR("Failed to resize image");
        free(resized_data);
        return false;
    }

    // Free old data and update image struct
    free(image.data);
    image.data = resized_data;
    image.width = final_width;
    image.height = final_height;

    return true;
}

sd_image_t ImageGenerator::createInitImage(const ImageGenerationParams& params) {
    sd_image_t init_image = base64ToImage(params.init_image_base64);
    if (!init_image.data) {
        throw std::runtime_error("Failed to decode init image");
    }

    if (!resizeImage(init_image, params.width, params.height)) {
        freeImage(init_image);
        throw std::runtime_error("Failed to resize init image");
    }

    return init_image;
}

sd_image_t ImageGenerator::createMaskImage(const ImageGenerationParams& params) {
    sd_image_t mask_image = {0, 0, 0, nullptr};

    if (params.mask_base64.empty()) {
        // Create a default mask for img2img (all white = no masking)
        mask_image.width = params.width;
        mask_image.height = params.height;
        mask_image.channel = 1;
        mask_image.data = static_cast<uint8_t*>(malloc(params.width * params.height * 1));
        if (!mask_image.data) {
            throw std::runtime_error("Failed to allocate mask image");
        }
        memset(mask_image.data, 255, params.width * params.height * 1);  // Fill with 255 (white)

        return mask_image;
    }

    // Decode provided mask
    mask_image = base64ToImage(params.mask_base64, 1);
    if (!mask_image.data) {
        throw std::runtime_error("Failed to decode mask image");
    }

    // Resize mask to match init image dimensions if needed
    if (mask_image.width != params.width || mask_image.height != params.height) {
        if (!resizeImage(mask_image, params.width, params.height)) {
            freeImage(mask_image);
            throw std::runtime_error("Failed to resize mask image");
        }
    }

    return mask_image;
}

void ImageGenerator::freeImage(sd_image_t& image) {
    if (image.data) {
        free(image.data);  // Works for both stbi allocated and manually allocated memory
        image.data = nullptr;
    }
    image.width = 0;
    image.height = 0;
    image.channel = 0;
}
bool ImageGenerator::needsModelReload(const std::string& model_path) const {
    // If not initialized, we need to load
    if (!initialized_ || !sd_ctx_) {
        return true;
    }

    // If model path changed, we need to reload
    if (model_path != current_model_path_) {
        return true;
    }

    return false;
}

bool ImageGenerator::ensureModelLoaded() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Get options
    auto options_wvalue = options_manager_->getOptions();
    std::string options_json = options_wvalue.dump();
    auto options = crow::json::load(options_json);

    if (!options) {
        LOG_ERROR("Failed to load options");
        return false;
    }

    // Get model path from options
    std::string model_path;
    if (options.has("sd_model_checkpoint")) {
        std::string model_name = std::string(options["sd_model_checkpoint"].s());

        if (!model_name.empty()) {
            // Get full path from model manager
            ModelInfo model_info = model_manager_->getModelByName(model_name, ModelType::CHECKPOINT);
            if (!model_info.full_path.empty()) {
                model_path = model_info.full_path;
            } else {
                LOG_ERROR("Model not found: %s", model_name.c_str());
                return false;
            }
        } else {
            LOG_ERROR("No model selected. Please configure sd_model_checkpoint in options.");
            return false;
        }
    } else {
        LOG_ERROR("No model selected. Please configure sd_model_checkpoint in options.");
        return false;
    }

    // Collect additional modules for comparison
    std::string vae_path_str;
    std::string clip_l_path_str;
    std::string clip_g_path_str;
    std::string t5xxl_path_str;

    if (options.has("forge_additional_modules")) {
        auto modules = options["forge_additional_modules"];
        if (modules.t() == crow::json::type::List) {
            for (size_t i = 0; i < modules.size(); i++) {
                std::string module_full_path = std::string(modules[i].s());
                if (module_full_path.empty()) {
                    continue;
                }

                LOG_DEBUG("Processing forge_additional_module: %s", module_full_path.c_str());

                // Inspect the model to determine its type
                std::string model_type = inspectModelType(module_full_path);

                if (model_type == "vae") {
                    vae_path_str = module_full_path;
                } else if (model_type == "clip_l") {
                    clip_l_path_str = module_full_path;
                } else if (model_type == "clip_g") {
                    clip_g_path_str = module_full_path;
                } else if (model_type == "t5xxl") {
                    t5xxl_path_str = module_full_path;
                } else {
                    LOG_WARNING("Unknown model type for: %s (detected as: %s)", module_full_path.c_str(),
                                model_type.c_str());
                }
            }
        }
    }

    // Check if we need to reload the model (check all paths)
    bool needs_reload = !initialized_ || !sd_ctx_ || model_path != current_model_path_ ||
                        vae_path_str != current_vae_path_ || clip_l_path_str != current_clip_l_path_ ||
                        clip_g_path_str != current_clip_g_path_ || t5xxl_path_str != current_t5xxl_path_;

    if (!needs_reload) {
        LOG_DEBUG("Model already loaded: %s", model_path.c_str());
        return true;
    }

    LOG_INFO("Model change detected, loading new model: %s", model_path.c_str());

    // Free old context if exists
    if (sd_ctx_) {
        LOG_INFO("Freeing old SD context for model switch");
        free_sd_ctx(sd_ctx_);
        sd_ctx_ = nullptr;
        initialized_ = false;
    }

    // Initialize with new model (mutex already held, don't call initialize which would deadlock)
    sd_set_log_callback(sd_log_cb, nullptr);

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
    params.vae_path = vae_path_str.empty() ? nullptr : vae_path_str.c_str();
    params.clip_l_path = clip_l_path_str.empty() ? nullptr : clip_l_path_str.c_str();
    params.clip_g_path = clip_g_path_str.empty() ? nullptr : clip_g_path_str.c_str();
    params.t5xxl_path = t5xxl_path_str.empty() ? nullptr : t5xxl_path_str.c_str();
    params.taesd_path = nullptr;
    params.lora_model_dir = nullptr;
    params.embedding_dir = nullptr;
    params.vae_decode_only = false;  // We need encoding for img2img

    // Log what we're loading
    if (!vae_path_str.empty()) {
        LOG_INFO("Loading VAE model: %s", vae_path_str.c_str());
    }
    if (!clip_l_path_str.empty()) {
        LOG_INFO("Loading CLIP-L model: %s", clip_l_path_str.c_str());
    }
    if (!clip_g_path_str.empty()) {
        LOG_INFO("Loading CLIP-G model: %s", clip_g_path_str.c_str());
    }
    if (!t5xxl_path_str.empty()) {
        LOG_INFO("Loading T5XXL model: %s", t5xxl_path_str.c_str());
    }

    // Set RNG type
    params.rng_type = CUDA_RNG;

    // Create SD context
    sd_ctx_ = new_sd_ctx(&params);
    if (!sd_ctx_) {
        LOG_ERROR("Failed to create SD context");
        return false;
    }

    // Track currently loaded model paths
    current_model_path_ = model_path;
    current_vae_path_ = vae_path_str;
    current_clip_l_path_ = clip_l_path_str;
    current_clip_g_path_ = clip_g_path_str;
    current_t5xxl_path_ = t5xxl_path_str;
    current_taesd_path_ = "";
    current_lora_model_dir_ = "";
    current_embeddings_dir_ = "";

    initialized_ = true;
    LOG_INFO("SD context initialized successfully");

    return true;
}
