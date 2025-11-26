#ifndef __IMAGE_GENERATOR_H__
#define __IMAGE_GENERATOR_H__

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "model_manager.h"
#include "options_manager.h"
#include "stable-diffusion.h"
#include "task_state.h"

struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int width = 512;
    int height = 512;
    int steps = 20;
    float cfg_scale = 7.0f;
    int64_t seed = -1;
    int batch_count = 1;
    int batch_size = 1;

    // Sampler settings
    sample_method_t sampler = EULER_A_SAMPLE_METHOD;
    scheduler_t scheduler = DISCRETE_SCHEDULER;

    // img2img specific
    std::string init_image_base64;
    std::string mask_base64;
    float strength = 0.75f;

    // Other options
    int clip_skip = -1;
};

class ImageGenerator {
   public:
    ImageGenerator(std::shared_ptr<TaskStateManager> task_state_manager,
                   std::shared_ptr<OptionsManager> options_manager, std::shared_ptr<ModelManager> model_manager);
    ~ImageGenerator();

    // Check if initialized
    bool isInitialized() const;

    // Get currently loaded model path
    std::string getCurrentModelPath() const;

    // Generate images
    std::vector<std::string> generateTxt2Img(const ImageGenerationParams& params, const std::string& task_id = "");

    std::vector<std::string> generateImg2Img(const ImageGenerationParams& params, const std::string& task_id = "");

    // Interrupt current generation
    void interrupt();

   private:
    // Convert sd_image_t to base64 string
    std::string imageToBase64(const sd_image_t& image);

    // Convert base64 string to sd_image_t
    sd_image_t base64ToImage(const std::string& base64_data, int desired_channels = 3);

    // Resize sd_image_t to target dimensions (modifies in place)
    bool resizeImage(sd_image_t& image, int target_width, int target_height);

    // Create init image from base64 string
    sd_image_t createInitImage(const ImageGenerationParams& params);

    // Create mask image from base64 string or default
    sd_image_t createMaskImage(const ImageGenerationParams& params);

    // Free sd_image_t data
    void freeImage(sd_image_t& image);

    // Generate image internally
    std::vector<std::string> generateInternal(const ImageGenerationParams& params, bool is_img2img,
                                              const std::string& task_id);

    // Check if model needs to be reloaded based on options
    bool needsModelReload(const std::string& model_path) const;

    // Ensure model is loaded based on current options
    bool ensureModelLoaded();

    sd_ctx_t* sd_ctx_;
    std::shared_ptr<TaskStateManager> task_state_manager_;
    std::shared_ptr<OptionsManager> options_manager_;
    std::shared_ptr<ModelManager> model_manager_;
    std::mutex mutex_;
    bool initialized_;
    bool interrupted_;
    std::string current_task_id_;

    // Track currently loaded model paths for change detection
    std::string current_model_path_;
    std::string current_vae_path_;
    std::string current_clip_l_path_;
    std::string current_clip_g_path_;
    std::string current_t5xxl_path_;
    std::string current_taesd_path_;
    std::string current_lora_model_dir_;
    std::string current_embeddings_dir_;
};

#endif  // __IMAGE_GENERATOR_H__
