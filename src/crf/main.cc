#include <iostream>
#include <map>
#include <chrono>
#include <ctime>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "logging.h"
#include "layer.h"
#include "training_utils.h"
#include "crf/crf.h"
#include "labeler/corpus.h"

namespace po = boost::program_options;

Corpus corpus;

void init_command_line(int argc, char* argv[], po::variables_map* conf) {
  po::options_description opts("LSTM-CRF");
  opts.add_options()
    ("train,t", "Should training be run?")
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The optimizer.")
    ("training_data,T", po::value<std::string>(), "The path to the training data.")
    ("devel_data,d", po::value<std::string>(), "The path to the development data.")
    ("pretrained,w", po::value<std::string>(), "The path to the pretrained word embedding.")
    ("model,m", po::value<std::string>(), "The path to the model, used for test pharse.")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy.")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probabilty with which to replace singletons.")
    ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
    ("word_dim", po::value<unsigned>()->default_value(32), "input embedding size")
    ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
    ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
    ("label_dim", po::value<unsigned>()->default_value(16), "label embedding size")
    ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
    ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
    ("maxiter", po::value<unsigned>()->default_value(10), "Max number of iterations.")
    ("conlleval", po::value<std::string>()->default_value("./conlleval.sh"), "config path to the conlleval script")
    ("dropout", po::value<float>()->default_value(0.), "the dropout rate")
    ("report_stops", po::value<unsigned>()->default_value(100), "the number of stops for reporting.")
    ("evaluate_stops", po::value<unsigned>()->default_value(2500), "the number of stops for evaluation.")
    ("verbose,v", "verbose log")
    ("help,h", "Show help information");

  po::store(po::parse_command_line(argc, argv, opts), *conf);
  if (conf->count("help")) {
    std::cerr << opts << std::endl;
    exit(1);
  }
  init_boost_log(conf->count("verbose"));
  if (conf->count("training_data") == 0) {
    _ERROR << "Please specify --training_data (-T): "
      << "this is required to determine the vocabulary mapping, even if the parser is used in prediction mode.";
    exit(1);
  }
}


std::string get_model_name(const po::variables_map& conf) {
  std::ostringstream os;
  os << "crf_" << conf["layers"].as<unsigned>()
    << "_" << conf["word_dim"].as<unsigned>()
    << "_" << conf["pos_dim"].as<unsigned>()
    << "_" << conf["lstm_input_dim"].as<unsigned>()
    << "_" << conf["hidden_dim"].as<unsigned>();
#ifndef _MSC_VER
  os << "-" << getpid() << ".params";
#endif
  return os.str();
}

double conlleval(const po::variables_map& conf,
  const std::string& tmp_output) {
#ifndef _MSC_VER
  std::string cmd = conf["conlleval"].as<std::string>() + " " + tmp_output;
  _TRACE << "Running: " << cmd << std::endl;
  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return 0.;
  }
  char buffer[128];
  std::string result = "";
  while (!feof(pipe)) {
    if (fgets(buffer, 128, pipe) != NULL) { result += buffer; }
  }
  pclose(pipe);

  std::stringstream S(result);
  std::string token;
  while (S >> token) {
    boost::algorithm::trim(token);
    return boost::lexical_cast<double>(token);
  }
#else
  return 1.;
#endif
  return 0.;
}


double evaluate(const po::variables_map& conf, CRFBuilder& engine, const std::string& tmp_output,
  const std::set<unsigned>& training_vocab) {
  auto kUNK = corpus.get_or_add_word(Corpus::UNK);
  double n_total = 0;

  auto t_start = std::chrono::high_resolution_clock::now();

  std::ofstream ofs(tmp_output);
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    const std::vector<unsigned>& raw_sentence = corpus.devel_sentences[sid];
    const std::vector<unsigned>& postag = corpus.devel_postags[sid];
    const std::vector<unsigned>& label = corpus.devel_labels[sid];
    const std::vector<std::string>& sentence_str = corpus.devel_sentences_str[sid];

    unsigned len = raw_sentence.size();

    std::vector<unsigned> sentence = raw_sentence;
    for (auto& w : sentence) {
      if (training_vocab.count(w) == 0) w = kUNK;
    }

    BOOST_ASSERT_MSG(len == postag.size(), "Unequal sentence and postag length");
    BOOST_ASSERT_MSG(len == label.size(), "Unequal sentence and gold label length");

    cnn::ComputationGraph hg;
    std::vector<unsigned> predict_label;
    engine.viterbi(&hg, raw_sentence, sentence, postag, predict_label);
    BOOST_ASSERT_MSG(len == predict_label.size(), "Unequal sentence and predict label length");
    
    n_total += label.size();
    for (unsigned i = 0; i < sentence.size(); ++i) {
      ofs << sentence_str[i] << " " <<
        corpus.id_to_postag[postag[i]] << " " <<
        corpus.id_to_label[label[i]] << " " <<
        corpus.id_to_label[predict_label[i]] << std::endl;
    }
    ofs << std::endl;
  }
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  double f_score = conlleval(conf, tmp_output);
  _INFO << "TEST f-score: " << f_score <<
    " [" << corpus.n_devel <<
    " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

void train(const po::variables_map& conf,
  cnn::Model& model,
  CRFBuilder& engine,
  const std::string& model_name,
  const std::string& tmp_output,
  const std::set<unsigned>& vocabulary,
  const std::set<unsigned>& singletons) {
  _INFO << "start training ...";

  // Setup the trainer.
  cnn::Trainer* trainer = get_trainer(conf, &model);

  // Order for shuffle.
  std::vector<unsigned> order(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { order[i] = i; }

  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);
  auto maxiter = conf["maxiter"].as<unsigned>();
  double n_seen = 0;
  double n_corr_tokens = 0, n_tokens = 0, llh = 0;
  double batchly_n_corr_tokens = 0, batchly_n_tokens = 0, batchly_llh = 0;
  /* # correct tokens, # tokens, # sentence, loglikelihood.*/

  int logc = 0;
  _INFO << "number of training instances: " << corpus.n_train;
  _INFO << "going to train " << maxiter << " iterations.";

  auto unk_strategy = conf["unk_strategy"].as<unsigned>();
  auto unk_prob = conf["unk_prob"].as<double>();
  double best_f_score = 0.;

  for (unsigned iter = 0; iter < maxiter; ++iter) {
    _INFO << "start of iteration #" << iter << ", training data is shuffled.";
    std::shuffle(order.begin(), order.end(), (*cnn::rndeng));

    for (unsigned i = 0; i < order.size(); ++i) {
      auto sid = order[i];
      const std::vector<unsigned>& raw_sentence = corpus.train_sentences[sid];
      const std::vector<unsigned>& postag = corpus.train_postags[sid];
      const std::vector<unsigned>& label = corpus.train_labels[sid];

      std::vector<unsigned> sentence = raw_sentence;
      if (unk_strategy == 1) {
        for (auto& w : sentence) {
          if (singletons.count(w) && cnn::rand01() < unk_prob) { w = kUNK; }
        }
      }

      double lp;
      {
        cnn::ComputationGraph hg;
        engine.supervised_loss(&hg, raw_sentence, sentence, postag, label);
        lp = cnn::as_scalar(hg.incremental_forward());
        BOOST_ASSERT_MSG(lp >= 0, "Log prob < 0 on sentence");
        hg.backward();
        trainer->update(1.);
      }

      llh += lp; batchly_llh += lp;
      n_seen += 1;
      ++logc;
      n_tokens += label.size(); batchly_n_tokens += label.size();

      if (logc % conf["report_stops"].as<unsigned>() == 0) {
        trainer->status();
        _INFO << "iter (batch) #" << iter << " (epoch " << n_seen / corpus.n_train
          << ") llh: " << batchly_llh << " ppl: " << exp(batchly_llh / batchly_n_tokens);
        n_corr_tokens += batchly_n_corr_tokens;
        batchly_llh = batchly_n_tokens = batchly_n_corr_tokens = 0.;
      }

      if (logc % conf["evaluate_stops"].as<unsigned>() == 0) {
        double f_score = evaluate(conf, engine, tmp_output, vocabulary);
        if (f_score > best_f_score) {
          best_f_score = f_score;
          _INFO << "new best record " << best_f_score << " is achieved, model updated.";
          std::ofstream out(model_name);
          boost::archive::text_oarchive oa(out);
          oa << model;
        }
      }
    }
    _INFO << "iteration #" << iter + 1 << " (epoch " << n_seen / corpus.n_train
      << ") llh: " << llh << " ppl: " << exp(llh / n_tokens);
    llh = n_tokens = n_corr_tokens = 0.;

    double f_score = evaluate(conf, engine, tmp_output, vocabulary);
    if (f_score > best_f_score) {
      best_f_score = f_score;
      _INFO << "new best record " << best_f_score << " is achieved, model updated.";
      std::ofstream out(model_name);
      boost::archive::text_oarchive oa(out);
      oa << model;
    }
    if (conf["optimizer"].as<std::string>() == "simple_sgd" || conf["optimizer"].as<std::string>() == "momentum_sgd") {
      trainer->update_epoch();
    }
  }

  delete trainer;
}

int main(int argc, char* argv[]) {
  cnn::Initialize(argc, argv, 1234);
  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) { std::cerr << ' ' << argv[i]; }
  std::cerr << std::endl;

  po::variables_map conf;
  init_command_line(argc, argv, &conf);

  if (conf["unk_strategy"].as<unsigned>() == 1) {
    _INFO << "unknown word strategy: STOCHASTIC REPLACEMENT"; 
  } else {
    _INFO << "unknown word strategy: NO REPLACEMENT";
  }

  std::string model_name;
  if (conf.count("train")) {
    model_name = get_model_name(conf);
    _INFO << "going to write parameters to file: " << model_name;
  } else {
    model_name = conf["model"].as<std::string>();
    _INFO << "going to load parameters from file: " << model_name;
  }

  corpus.load_training_data(conf["training_data"].as<std::string>());

  std::set<unsigned> training_vocab, singletons;
  corpus.get_vocabulary_and_singletons(training_vocab, singletons);

  std::unordered_map<unsigned, std::vector<float>> pretrained;
  if (conf.count("pretrained")) {
    load_pretrained_word_embedding(conf["pretrained"].as<std::string>(),
      conf["pretrained_dim"].as<unsigned>(), pretrained, corpus);
    _INFO << "pretrained word embedding is loaded.";
  }

  cnn::Model model;
  CRFBuilder crf(&model,
    corpus.max_word + 1, conf["word_dim"].as<unsigned>(),
    corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
    corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
    corpus.n_labels, conf["label_dim"].as<unsigned>(),
    conf["layers"].as<unsigned>(), conf["lstm_input_dim"].as<unsigned>(),
    conf["hidden_dim"].as<unsigned>(),
    conf["dropout"].as<float>(),
    pretrained);

  corpus.load_devel_data(conf["devel_data"].as<std::string>());
#ifdef _MSC_VER
  std::string tmp_output = "crf.evaluator";
#else
  std::string tmp_output = "/tmp/crf.evaluator." + boost::lexical_cast<std::string>(getpid());
#endif

  if (conf.count("train")) {
    train(conf, model, crf, model_name, tmp_output, training_vocab, singletons);
  }

  std::ifstream in(model_name);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  evaluate(conf, crf, tmp_output, training_vocab);
  return 0;
}
