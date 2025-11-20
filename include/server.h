#ifndef __SERVER_H__
#define __SERVER_H__

#include <memory>
#include <string>

#include "crow.h"
#include "options_manager.h"
#include "task_state.h"

class Server {
   public:
    Server(int port = 8188);
    ~Server();

    void run();
    void stop();

   private:
    void setupRoutes();

    // Route handlers
    crow::response handlePing();
    crow::response handleGetOptions();
    crow::response handlePostOptions(const crow::request& req);
    crow::response handleTxt2Img(const crow::request& req);
    crow::response handleImg2Img(const crow::request& req);
    crow::response handleProgress(const crow::request& req);
    crow::response handleInterrupt(const crow::request& req);
    crow::response handleExtraBatchImages(const crow::request& req);
    crow::response handleControlNetDetect(const crow::request& req);
    crow::response handleRefreshCheckpoints();
    crow::response handleRefreshVaeAndTextEncoders();

    // Helper methods
    crow::response generateImage(const crow::json::rvalue& json_body, bool is_img2img);

    int port_;
    crow::SimpleApp app_;
    std::unique_ptr<OptionsManager> options_manager_;
    std::unique_ptr<TaskStateManager> task_state_manager_;
    bool should_stop_;
};

#endif  // __SERVER_H__
