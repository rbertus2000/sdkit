#include "server.h"

#include <chrono>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "logging.h"

// Custom log handler for Crow that filters out /ping requests
class FilteredLogHandler : public crow::ILogHandler {
   public:
    void log(std::string message, crow::LogLevel level) override {
        // Skip logging /ping requests to reduce noise
        if (message.find("/ping") != std::string::npos || message.find("/internal/progress") != std::string::npos) {
            return;
        }

        // Use our existing logging system for consistency
        LogLevel our_level;
        switch (level) {
            case crow::LogLevel::Debug:
                our_level = LogLevel::Debug;
                break;
            case crow::LogLevel::Info:
                our_level = LogLevel::Info;
                break;
            case crow::LogLevel::Warning:
                our_level = LogLevel::Warning;
                break;
            case crow::LogLevel::Error:
            case crow::LogLevel::Critical:
                our_level = LogLevel::Error;
                break;
            default:
                our_level = LogLevel::Info;
                break;
        }

        log_message(our_level, "[CROW] %s", message.c_str());
    }
};

// Convert webui (Forge/Automatic1111 style) sampler/scheduler names
// into stable-diffusion.cpp compatible names.
static std::string convert_webui_sampler_name(const std::string& name) {
    static const std::unordered_map<std::string, std::string> mapping = {
        {"Euler", "euler"},
        {"Euler a", "euler_a"},
        {"Heun", "heun"},
        {"DPM2", "dpm2"},
        {"DPM++ 2S a", "dpm++2s_a"},
        {"DPM++ 2M", "dpm++2m"},
        {"DPM++ 2M v2", "dpm++2mv2"},
        {"IPNDM", "ipndm"},
        {"IPNDM_V", "ipndm_v"},
        {"LCM", "lcm"},
        {"DDIM", "ddim_trailing"},
        {"TCD", "tcd"},
    };

    auto it = mapping.find(name);
    if (it != mapping.end()) return it->second;
    return name;
}

static std::string convert_webui_scheduler_name(const std::string& name) {
    static const std::unordered_map<std::string, std::string> mapping = {
        {"automatic", "discrete"},      {"uniform", "discrete"},           {"karras", "karras"},
        {"exponential", "exponential"}, {"sgm_uniform", "sgm_uniform"},    {"simple", "simple"},
        {"align_your_steps", "ays"},    {"align_your_steps_GITS", "gits"},
    };

    auto it = mapping.find(name);
    if (it != mapping.end()) return it->second;
    return name;
}

Server::Server(int port, std::shared_ptr<ModelManager> model_manager)
    : port_(port), model_manager_(model_manager), should_stop_(false) {
    // Set up custom logger to filter out unnecessary requests
    static FilteredLogHandler filtered_handler;
    crow::logger::setHandler(&filtered_handler);

    options_manager_ = std::make_shared<OptionsManager>();
    task_state_manager_ = std::make_shared<TaskStateManager>();

    // Create ImageGenerator with shared task state manager and model manager
    image_generator_ = std::make_unique<ImageGenerator>(task_state_manager_, options_manager_, model_manager_);

    // Load existing options
    options_manager_->load();

    setupRoutes();
}

Server::~Server() { stop(); }

void Server::setupRoutes() {
    // Ping endpoint
    CROW_ROUTE(app_, "/v1/internal/ping").methods("GET"_method)([this]() { return handlePing(); });

    // Options endpoints
    CROW_ROUTE(app_, "/v1/sdapi/v1/options").methods("GET"_method)([this]() { return handleGetOptions(); });

    CROW_ROUTE(app_, "/v1/sdapi/v1/options").methods("POST"_method)([this](const crow::request& req) {
        return handlePostOptions(req);
    });

    // Image generation endpoints
    CROW_ROUTE(app_, "/v1/sdapi/v1/txt2img").methods("POST"_method)([this](const crow::request& req) {
        return handleTxt2Img(req);
    });

    CROW_ROUTE(app_, "/v1/sdapi/v1/img2img").methods("POST"_method)([this](const crow::request& req) {
        return handleImg2Img(req);
    });

    // Progress endpoint
    CROW_ROUTE(app_, "/v1/internal/progress").methods("POST"_method)([this](const crow::request& req) {
        return handleProgress(req);
    });

    // Interrupt endpoint
    CROW_ROUTE(app_, "/v1/sdapi/v1/interrupt").methods("POST"_method)([this](const crow::request& req) {
        return handleInterrupt(req);
    });

    // Extra batch images endpoint
    CROW_ROUTE(app_, "/v1/sdapi/v1/extra-batch-images").methods("POST"_method)([this](const crow::request& req) {
        return handleExtraBatchImages(req);
    });

    // ControlNet detect endpoint
    CROW_ROUTE(app_, "/v1/controlnet/detect").methods("POST"_method)([this](const crow::request& req) {
        return handleControlNetDetect(req);
    });

    // Refresh endpoints
    CROW_ROUTE(app_, "/v1/sdapi/v1/refresh-checkpoints").methods("POST"_method)([this]() {
        return handleRefreshCheckpoints();
    });

    CROW_ROUTE(app_, "/v1/sdapi/v1/refresh-vae-and-text-encoders").methods("POST"_method)([this]() {
        return handleRefreshVaeAndTextEncoders();
    });
}

void Server::run() {
    std::cout << "Starting server on port " << port_ << std::endl;
    app_.port(port_).multithreaded().run();
}

void Server::stop() {
    should_stop_ = true;
    app_.stop();
}

crow::response Server::handlePing() { return crow::response(200, "OK"); }

crow::response Server::handleGetOptions() {
    try {
        auto options = options_manager_->getOptions();
        return crow::response(200, options);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to get options: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handlePostOptions(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["error"] = "Invalid JSON";
            return crow::response(400, error);
        }

        if (options_manager_->setOptions(json_body)) {
            return crow::response(200, "OK");
        } else {
            crow::json::wvalue error;
            error["error"] = "Failed to save options";
            return crow::response(500, error);
        }
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to set options: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleTxt2Img(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["error"] = "Invalid JSON";
            return crow::response(400, error);
        }

        return generateImage(json_body, false);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to generate image: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleImg2Img(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["error"] = "Invalid JSON";
            return crow::response(400, error);
        }

        return generateImage(json_body, true);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to generate image: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::generateImage(const crow::json::rvalue& json_body, bool is_img2img) {
    // Extract task_id
    std::string task_id = "default_task";
    if (json_body.has("force_task_id")) {
        task_id = json_body["force_task_id"].s();
    }

    // Create task
    task_state_manager_->createTask(task_id);

    try {
        // Parse generation parameters
        ImageGenerationParams params;
        params.prompt = json_body.has("prompt") ? std::string(json_body["prompt"].s()) : "";
        params.negative_prompt = json_body.has("negative_prompt") ? std::string(json_body["negative_prompt"].s()) : "";
        params.width = json_body.has("width") ? json_body["width"].i() : 512;
        params.height = json_body.has("height") ? json_body["height"].i() : 512;
        params.steps = json_body.has("steps") ? json_body["steps"].i() : 20;
        params.cfg_scale = json_body.has("cfg_scale") ? json_body["cfg_scale"].d() : 7.0f;
        params.seed = json_body.has("seed") ? json_body["seed"].i() : -1;
        params.batch_count = json_body.has("batch_size") ? json_body["batch_size"].i() : 1;

        // Sampler and scheduler parameters
        if (json_body.has("sampler_name")) {
            std::string sampler_str = json_body["sampler_name"].s();
            // Convert from webui-style sampler name to sd.cpp name
            sampler_str = convert_webui_sampler_name(sampler_str);
            params.sampler = str_to_sample_method(sampler_str.c_str());
        }
        if (json_body.has("scheduler")) {
            std::string scheduler_str = json_body["scheduler"].s();
            // Convert from webui-style scheduler name to sd.cpp name
            scheduler_str = convert_webui_scheduler_name(scheduler_str);
            params.scheduler = str_to_scheduler(scheduler_str.c_str());
        }

        // img2img specific parameters
        if (is_img2img) {
            if (json_body.has("init_images") && json_body["init_images"].size() > 0) {
                params.init_image_base64 = std::string(json_body["init_images"][0].s());
            }
            if (json_body.has("mask")) {
                params.mask_base64 = std::string(json_body["mask"].s());
            }
            params.strength = json_body.has("denoising_strength") ? json_body["denoising_strength"].d() : 0.75f;
        }

        // Generate images (runs in same thread, blocks until complete)
        std::vector<std::string> images;
        if (is_img2img) {
            images = image_generator_->generateImg2Img(params, task_id);
        } else {
            images = image_generator_->generateTxt2Img(params, task_id);
        }

        // Create info JSON string
        crow::json::wvalue info_json;
        info_json["prompt"] = params.prompt;
        info_json["negative_prompt"] = params.negative_prompt;
        info_json["steps"] = params.steps;
        info_json["cfg_scale"] = params.cfg_scale;
        info_json["seed"] = params.seed;
        info_json["width"] = params.width;
        info_json["height"] = params.height;

        crow::json::wvalue infotexts_json;
        infotexts_json["infotexts"] = info_json.dump();
        std::string info = infotexts_json.dump();

        // Complete task
        task_state_manager_->completeTask(task_id, images, info);

        // Return response
        crow::json::wvalue response;
        response["images"] = images;
        response["info"] = info;

        return crow::response(200, response);

    } catch (const std::exception& e) {
        LOG_ERROR("Image generation error: %s", e.what());
        crow::json::wvalue error;
        error["error"] = std::string("Generation failed: ") + e.what();
        task_state_manager_->completeTask(task_id, {}, error.dump());
        return crow::response(500, error);
    }
}

crow::response Server::handleProgress(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["error"] = "Invalid JSON";
            return crow::response(400, error);
        }

        if (!json_body.has("id_task")) {
            crow::json::wvalue error;
            error["error"] = "Missing id_task parameter";
            return crow::response(400, error);
        }

        std::string task_id = json_body["id_task"].s();

        if (!task_state_manager_->taskExists(task_id)) {
            crow::json::wvalue error;
            error["error"] = "Task not found";
            return crow::response(404, error);
        }

        TaskState state = task_state_manager_->getTaskState(task_id);

        crow::json::wvalue response;
        response["completed"] = state.completed;
        response["progress"] = state.progress;
        response["live_preview"] = state.live_preview;
        response["id_live_preview"] = state.id_live_preview;

        return crow::response(200, response);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to get progress: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleInterrupt(const crow::request& req) {
    try {
        // Interrupt the image generator
        if (image_generator_) {
            image_generator_->interrupt();
            LOG_INFO("Image generation interrupted");
        }

        return crow::response(200, "OK");
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to interrupt: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleExtraBatchImages(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["error"] = "Invalid JSON";
            return crow::response(400, error);
        }

        // Stub implementation: just return processed versions of input images
        std::vector<std::string> result_images;

        if (json_body.has("imageList")) {
            auto image_list = json_body["imageList"];
            for (size_t i = 0; i < image_list.size(); i++) {
                // In real implementation, apply upscaling/face restoration
                result_images.push_back("upscaled_image_base64_" + std::to_string(i));
            }
        }

        crow::json::wvalue response;
        response["images"] = result_images;

        return crow::response(200, response);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to process images: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleControlNetDetect(const crow::request& req) {
    try {
        auto json_body = crow::json::load(req.body);
        if (!json_body) {
            crow::json::wvalue error;
            error["error"] = "Invalid JSON";
            return crow::response(400, error);
        }

        // Stub implementation: return processed versions of input images
        std::vector<std::string> result_images;

        if (json_body.has("controlnet_input_images")) {
            auto input_images = json_body["controlnet_input_images"];
            std::string module = "canny";
            if (json_body.has("controlnet_module")) {
                module = json_body["controlnet_module"].s();
            }
            for (size_t i = 0; i < input_images.size(); i++) {
                // In real implementation, apply controlnet detection
                result_images.push_back("detected_" + module + "_base64_" + std::to_string(i));
            }
        }

        crow::json::wvalue response;
        response["images"] = result_images;

        return crow::response(200, response);
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to detect: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleRefreshCheckpoints() {
    try {
        LOG_INFO("Refreshing checkpoints...");
        if (model_manager_) {
            model_manager_->refreshCheckpoints();
        }
        return crow::response(200, "OK");
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to refresh checkpoints: ") + e.what();
        return crow::response(500, error);
    }
}

crow::response Server::handleRefreshVaeAndTextEncoders() {
    try {
        LOG_INFO("Refreshing VAE and text encoders...");
        if (model_manager_) {
            model_manager_->refreshVaeAndTextEncoders();
        }
        return crow::response(200, "OK");
    } catch (const std::exception& e) {
        crow::json::wvalue error;
        error["error"] = std::string("Failed to refresh VAE and text encoders: ") + e.what();
        return crow::response(500, error);
    }
}
