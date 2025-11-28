#include <csignal>
#include <iostream>
#include <memory>
#include <string>

#include "logging.h"
#include "model_manager.h"
#include "server.h"

std::unique_ptr<Server> g_server;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

struct CommandLineArgs {
    int port = 8188;
    std::string log_level = "info";
    int parent_pid = 0;
    std::string ckpt_dir;
    std::string vae_dir;
    std::string hypernetwork_dir;
    std::string gfpgan_models_path;
    std::string realesrgan_models_path;
    std::string lora_dir;
    std::string codeformer_models_path;
    std::string embeddings_dir;
    std::string controlnet_dir;
    std::string text_encoder_dir;
};

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --port <port>                      Server port (default: 8188)" << std::endl;
    std::cerr << "  --log-level <level>                Log level: debug, info, warning, error (default: info)"
              << std::endl;
    std::cerr << "  --parent-pid <pid>                 Parent process PID" << std::endl;
    std::cerr << "  --ckpt-dir <path>                  Checkpoint models directory" << std::endl;
    std::cerr << "  --vae-dir <path>                   VAE models directory" << std::endl;
    std::cerr << "  --hypernetwork-dir <path>          Hypernetwork models directory" << std::endl;
    std::cerr << "  --gfpgan-models-path <path>        GFPGAN models directory" << std::endl;
    std::cerr << "  --realesrgan-models-path <path>    RealESRGAN models directory" << std::endl;
    std::cerr << "  --lora-dir <path>                  LoRA models directory" << std::endl;
    std::cerr << "  --codeformer-models-path <path>    Codeformer models directory" << std::endl;
    std::cerr << "  --embeddings-dir <path>            Embeddings directory" << std::endl;
    std::cerr << "  --controlnet-dir <path>            ControlNet models directory" << std::endl;
    std::cerr << "  --text-encoder-dir <path>          Text encoder models directory" << std::endl;
}

CommandLineArgs parse_args(int argc, char* argv[]) {
    CommandLineArgs args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--port" && i + 1 < argc) {
            try {
                args.port = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid port number: " << argv[i] << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        } else if (arg == "--log-level" && i + 1 < argc) {
            args.log_level = argv[++i];
        } else if (arg == "--parent-pid" && i + 1 < argc) {
            try {
                args.parent_pid = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                std::cerr << "Invalid parent PID: " << argv[i] << std::endl;
                print_usage(argv[0]);
                exit(1);
            }
        } else if (arg == "--ckpt-dir" && i + 1 < argc) {
            args.ckpt_dir = argv[++i];
        } else if (arg == "--vae-dir" && i + 1 < argc) {
            args.vae_dir = argv[++i];
        } else if (arg == "--hypernetwork-dir" && i + 1 < argc) {
            args.hypernetwork_dir = argv[++i];
        } else if (arg == "--gfpgan-models-path" && i + 1 < argc) {
            args.gfpgan_models_path = argv[++i];
        } else if (arg == "--realesrgan-models-path" && i + 1 < argc) {
            args.realesrgan_models_path = argv[++i];
        } else if (arg == "--lora-dir" && i + 1 < argc) {
            args.lora_dir = argv[++i];
        } else if (arg == "--codeformer-models-path" && i + 1 < argc) {
            args.codeformer_models_path = argv[++i];
        } else if (arg == "--embeddings-dir" && i + 1 < argc) {
            args.embeddings_dir = argv[++i];
        } else if (arg == "--controlnet-dir" && i + 1 < argc) {
            args.controlnet_dir = argv[++i];
        } else if (arg == "--text-encoder-dir" && i + 1 < argc) {
            args.text_encoder_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Parse command line arguments
    CommandLineArgs args = parse_args(argc, argv);

    // Set log level from command line argument
    set_log_level(args.log_level);

    std::cout << "==================================" << std::endl;
    std::cout << "  Stable Diffusion API Server" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Port: " << args.port << std::endl;
    std::cout << std::endl;

    // Create and configure model manager
    auto model_manager = std::make_shared<ModelManager>();

    if (!args.ckpt_dir.empty()) {
        model_manager->setCheckpointDir(args.ckpt_dir);
    }
    if (!args.vae_dir.empty()) {
        model_manager->setVaeDir(args.vae_dir);
    }
    if (!args.hypernetwork_dir.empty()) {
        model_manager->setHypernetworkDir(args.hypernetwork_dir);
    }
    if (!args.gfpgan_models_path.empty()) {
        model_manager->setGfpganModelsPath(args.gfpgan_models_path);
    }
    if (!args.realesrgan_models_path.empty()) {
        model_manager->setRealesrganModelsPath(args.realesrgan_models_path);
    }
    if (!args.lora_dir.empty()) {
        model_manager->setLoraDir(args.lora_dir);
    }
    if (!args.codeformer_models_path.empty()) {
        model_manager->setCodeformerModelsPath(args.codeformer_models_path);
    }
    if (!args.embeddings_dir.empty()) {
        model_manager->setEmbeddingsDir(args.embeddings_dir);
    }
    if (!args.controlnet_dir.empty()) {
        model_manager->setControlnetDir(args.controlnet_dir);
    }
    if (!args.text_encoder_dir.empty()) {
        model_manager->setTextEncoderDir(args.text_encoder_dir);
    }

    // Scan all model directories
    std::cout << "Scanning model directories..." << std::endl;
    model_manager->scanAllDirectories();
    std::cout << "Model scanning complete." << std::endl;
    std::cout << std::endl;

    try {
        // Create and start the server
        g_server = std::make_unique<Server>(args.port, model_manager);
        g_server->run();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Server stopped." << std::endl;
    return 0;
}
