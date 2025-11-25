#ifndef __IMAGE_GENERATOR_H__
#define __IMAGE_GENERATOR_H__

#include <memory>
#include <mutex>
#include <string>
#include <vector>

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
    float strength = 0.75f;

    // Other options
    int clip_skip = -1;
};

class ImageGenerator {
   public:
    ImageGenerator(std::shared_ptr<TaskStateManager> task_state_manager,
                   std::shared_ptr<OptionsManager> options_manager);
    ~ImageGenerator();

    // Initialize SD context
    bool initialize(const std::string& model_path, const std::string& vae_path = "", const std::string& taesd_path = "",
                    const std::string& lora_model_dir = "", const std::string& embeddings_dir = "", int n_threads = -1,
                    sd_type_t wtype = SD_TYPE_COUNT, bool offload_to_cpu = false, bool vae_on_cpu = false);

    // Check if initialized
    bool isInitialized() const;

    // Generate images
    std::vector<std::string> generateTxt2Img(const ImageGenerationParams& params, const std::string& task_id = "");

    std::vector<std::string> generateImg2Img(const ImageGenerationParams& params, const std::string& task_id = "");

    // Interrupt current generation
    void interrupt();

   private:
    // Convert sd_image_t to base64 string
    std::string imageToBase64(const sd_image_t& image);

    // Convert base64 string to sd_image_t
    sd_image_t base64ToImage(const std::string& base64_data);

    // Free sd_image_t data
    void freeImage(sd_image_t& image);

    // Generate image internally
    std::vector<std::string> generateInternal(const ImageGenerationParams& params, bool is_img2img,
                                              const std::string& task_id);

    sd_ctx_t* sd_ctx_;
    std::shared_ptr<TaskStateManager> task_state_manager_;
    std::shared_ptr<OptionsManager> options_manager_;
    std::mutex mutex_;
    bool initialized_;
    bool interrupted_;
    std::string current_task_id_;
};

#endif  // __IMAGE_GENERATOR_H__
