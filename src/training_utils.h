#ifndef __TRAINING_UTILS_H__
#define __TRAINING_UTILS_H__

#include "cnn/model.h"
#include "cnn/training.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

cnn::Trainer* get_trainer(const po::variables_map& conf, cnn::Model* model);

#endif  //  end for __TRAINING_UTILS_H__
