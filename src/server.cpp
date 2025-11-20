#include "server.h"

#include <chrono>
#include <sstream>
#include <thread>

#include "logging.h"

Server::Server(int port, std::shared_ptr<ModelManager> model_manager)
    : port_(port), model_manager_(model_manager), should_stop_(false) {
    options_manager_ = std::make_unique<OptionsManager>();
    task_state_manager_ = std::make_unique<TaskStateManager>();

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

    // Simulate image generation with progress updates
    // In a real implementation, this would call the actual stable-diffusion logic
    int steps = json_body.has("steps") ? json_body["steps"].i() : 25;
    int batch_size = json_body.has("batch_size") ? json_body["batch_size"].i() : 1;

    for (int i = 0; i <= steps; i++) {
        float progress = static_cast<float>(i) / steps;
        std::string live_preview = "preview_image_base64_" + std::to_string(i);
        task_state_manager_->updateTaskProgress(task_id, progress, live_preview);

        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Generate result images (stub data)
    std::vector<std::string> images;
    for (int i = 0; i < batch_size; i++) {
        images.push_back("generated_image_base64_" + std::to_string(i));
    }

    // Create info JSON string
    std::string info = "{\"infotexts\": \"Generated image info\"}";

    // Complete task
    task_state_manager_->completeTask(task_id, images, info);

    // Return response
    crow::json::wvalue response;
    response["images"] = images;
    response["info"] = info;

    return crow::response(200, response);
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
        // In a real implementation, this would signal the generation thread to stop
        // For now, we'll just mark all active tasks as interrupted

        // This is a stub - in reality you'd need to track the current task
        // and signal it to stop

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
