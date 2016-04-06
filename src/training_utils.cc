#include <cstdio>
#include "training_utils.h"
#include "logging.h"


cnn::Trainer* get_trainer(const po::variables_map& conf, cnn::Model* model) {
  cnn::Trainer* trainer = nullptr;
  if (conf["optimizer"].as<std::string>() == "simple_sgd") {
    trainer = new cnn::SimpleSGDTrainer(model);
    trainer->eta_decay = 0.08f;
    if (conf.count("eta")) { trainer->eta = conf["eta"].as<float>();  }
  } else if (conf["optimizer"].as<std::string>() == "momentum_sgd") {
    trainer = new cnn::MomentumSGDTrainer(model);
    trainer->eta_decay = 0.08f;
    if (conf.count("eta")) { trainer->eta = conf["eta"].as<float>();  }
  } else if (conf["optimizer"].as<std::string>() == "adagrad") {
    trainer = new cnn::AdagradTrainer(model);
    trainer->eta = 0.01f;
    trainer->clipping_enabled = false;
    static_cast<cnn::AdagradTrainer*>(trainer)->epsilon = 1e-6f;
  } else if (conf["optimizer"].as<std::string>() == "adadelta") {
    trainer = new cnn::AdadeltaTrainer(model);
    trainer->clipping_enabled = false;
  } else if (conf["optimizer"].as<std::string>() == "rmsprop") {
    trainer = new cnn::RmsPropTrainer(model);
  } else if (conf["optimizer"].as<std::string>() == "adam") {
    trainer = new cnn::AdamTrainer(model);
    // Adam parameter is obtained from Kong
    trainer->eta = 0.0005f;
    static_cast<cnn::AdamTrainer*>(trainer)->beta_1 = 0.01f;
    trainer->clipping_enabled = false;
  } else {
    _ERROR << "Unknown optimizer type.";
    exit(1);
  }
  return trainer;
}


