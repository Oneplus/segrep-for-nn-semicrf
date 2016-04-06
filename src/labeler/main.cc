#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <ctime>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif
#include <signal.h>

#include "utils.h"
#include "logging.h"
#include "training_utils.h"
#include "labeler/corpus.h"
#include "labeler/model.h"
#include "cnn/training.h"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

namespace po = boost::program_options;

Corpus corpus = Corpus();

void init_command_line(int argc, char* argv[], po::variables_map* conf) {
  po::options_description opts("Configuration options");
  opts.add_options()
    ("graph", po::value<std::string>()->default_value("lstm"),
    "The type of graph, avaliable [lstm, bilstm, bilstm_glob, bilstm_out_seq, bilstm_out_seq_att, bilstm_to_seq, bilstm_to_seq_w, bilstm_to_seq_att].")
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The optimizer.")
    ("training_data,T", po::value<std::string>(), "Training corpus.")
    ("dev_data,d", po::value<std::string>(), "Development corpus")
    ("test_data,p", po::value<std::string>(), "Test corpus")
    ("pretrained,w", po::value<std::string>(), "Pretrained word embeddings")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy: 1 = singletons become UNK with probability unk_prob")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probably with which to replace singletons with UNK in training data")
    ("model,m", po::value<std::string>(), "Load saved model from this file")
    ("train,t", "Should training be run?")
    ("use_postags,P",  "make POS tags visible to parser")
    ("layers", po::value<unsigned>()->default_value(2), "number of LSTM layers")
    ("word_dim", po::value<unsigned>()->default_value(32), "input embedding size")
    ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
    ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
    ("hidden_dim", po::value<unsigned>()->default_value(64), "hidden dimension")
    ("attention_window_size", po::value<unsigned>()->default_value(5), "the window size for attention [used when graph=chris_attention]")
    ("lstm_input_dim", po::value<unsigned>()->default_value(60), "LSTM input dimension")
    ("label_dim", po::value<unsigned>()->default_value(16), "label embedding size")
    ("maxiter", po::value<unsigned>()->default_value(10), "Max number of iterations.")
    ("conlleval", po::value<std::string>()->default_value("./conlleval.sh"), "config path to the conlleval script")
    ("report_stops", po::value<unsigned>()->default_value(100), "the number of stops for reporting.")
    ("evaluate_stops", po::value<unsigned>()->default_value(2500), "the number of stops for evaluation.")
    ("verbose,v", "verbose log")
    ("help,h", "Help");

  po::options_description dcmdline_options;
  dcmdline_options.add(opts);
  po::store(parse_command_line(argc, argv, dcmdline_options), *conf);

  if (conf->count("help")) {
    std::cerr << dcmdline_options << std::endl;
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
  os << "labeler_" << conf["graph"].as<std::string>()
    << "_" << (conf.count("use_postags") ? "pos" : "nopos")
    << "_" << conf["layers"].as<unsigned>()
    << "_" << conf["word_dim"].as<unsigned>()
    << "_" << conf["pos_dim"].as<unsigned>()
    << "_" << conf["label_dim"].as<unsigned>()
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

double evaluate(const po::variables_map& conf,
  SequenceLabelingModel& engine,
  const std::string& tmp_output,
  const std::set<unsigned>& training_vocab) {
  auto kUNK = corpus.get_or_add_word(Corpus::UNK);
  double n_total = 0;

  auto t_start = std::chrono::high_resolution_clock::now();
  double dummy;

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
    engine.log_probability(&hg,
      raw_sentence, sentence, sentence_str, postag, std::vector<unsigned>(),
      dummy, predict_label);
    
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


template<typename EngineType> EngineType* get_engine(const po::variables_map& conf,
  cnn::Model* model, const Corpus& corpus,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained) {
  return new EngineType(model,
    corpus.max_word + 1,
    conf["word_dim"].as<unsigned>(),
    corpus.max_postag + 10,
    conf["pos_dim"].as<unsigned>(),
    corpus.max_word + 1,
    conf["pretrained_dim"].as<unsigned>(),
    conf["lstm_input_dim"].as<unsigned>(),
    conf["layers"].as<unsigned>(),
    conf["hidden_dim"].as<unsigned>(),
    corpus.n_labels,
    pretrained,
    corpus.id_to_label);
}


template<typename EngineType> EngineType* get_engine2(const po::variables_map& conf,
  cnn::Model* model, const Corpus& corpus,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained) {
  return new EngineType(model,
    corpus.max_word + 1,
    conf["word_dim"].as<unsigned>(),
    corpus.max_postag + 10,
    conf["pos_dim"].as<unsigned>(),
    corpus.max_word + 1,
    conf["pretrained_dim"].as<unsigned>(),
    conf["lstm_input_dim"].as<unsigned>(),
    conf["layers"].as<unsigned>(),
    conf["hidden_dim"].as<unsigned>(),
    corpus.n_labels,
    conf["label_dim"].as<unsigned>(),
    pretrained,
    corpus.id_to_label);
}


void train(const po::variables_map& conf,
  cnn::Model& model,
  SequenceLabelingModel& engine,
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
    // Use cnn::random generator to guarentee same result across different machine.
    std::shuffle(order.begin(), order.end(), (*cnn::rndeng));

    for (unsigned i = 0; i < order.size(); ++i) {
      auto sid = order[i];
      const std::vector<unsigned>& raw_sentence = corpus.train_sentences[sid];
      const std::vector<unsigned>& postag = corpus.train_postags[sid];
      const std::vector<unsigned>& label = corpus.train_labels[sid];

      std::vector<unsigned> sentence = raw_sentence;
      if (unk_strategy == 1) {
        for (auto& w : sentence) {
          if (singletons.count(w) && cnn::rand01() < unk_prob) { w = kUNK;  }
        }
      }

      double lp;
      {
        cnn::ComputationGraph hg;
        std::vector<unsigned> predict_labels;
        engine.log_probability(&hg, raw_sentence, sentence, std::vector<std::string>(),
            postag, label, batchly_n_corr_tokens, predict_labels);

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
          << ") llh: " << batchly_llh << " ppl: " << exp(batchly_llh / batchly_n_tokens)
          << " err: " << (batchly_n_tokens - batchly_n_corr_tokens) / batchly_n_tokens;
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
      << ") llh: " << llh << " ppl: " << exp(llh / n_tokens)
      << " err: " << (n_tokens - n_corr_tokens) / n_tokens;
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
  /* NOTE! The third argument specify the random seed which is configed in
   * initialization function */
  cnn::Initialize(argc, argv, 1234);
  std::cerr << "command:";
  for (unsigned i = 0; i < static_cast<unsigned>(argc); ++i) std::cerr << ' ' << argv[i];
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
    _INFO << "pre-trained word embedding is loaded.";
  }

  cnn::Model model;
  SequenceLabelingModel* engine = nullptr;
  _INFO << "building " << conf["graph"].as<std::string>();
  if (conf["graph"].as<std::string>() == "lstm") {
    engine = get_engine<LSTMNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "revlstm_to_seq") {
    engine = get_engine2<RevLSTM2SeqNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "bilstm") {
    engine = get_engine<BiLSTMNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "bilstm_glob") {
    engine = get_engine<GlobBiLSTMNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "bilstm_out_seq") {
    engine = get_engine<LSTMBiLSTMNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "bilstm_out_seq_att") {
    engine = new LSTMAttendBiLSTMNERLabeler(&model,
      corpus.max_word + 1,
      conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10,
      conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1,
      conf["pretrained_dim"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["layers"].as<unsigned>(),
      conf["hidden_dim"].as<unsigned>(),
      corpus.n_labels,
      conf["attention_window_size"].as<unsigned>(),
      training_vocab,
      pretrained,
      corpus.id_to_label);
  } else if (conf["graph"].as<std::string>() == "bilstm_to_seq") {
    engine = get_engine2<Seq2SeqLabelVerBiLSTMNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "bilstm_to_seq_w") {
    engine = get_engine2<Seq2SeqWordVerBiLSTMNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "bilstm_to_seq_att") {
    engine = get_engine2<Seq2SeqAttentionVerLSTMNERLabeler>(conf, &model, corpus, pretrained);
  } else if (conf["graph"].as<std::string>() == "single") {
    engine = new BiLSTMNERLabeler(&model, corpus.max_word + 1, conf["word_dim"].as<unsigned>(),
        0, 0,
        corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
        conf["lstm_input_dim"].as<unsigned>(),
        conf["layers"].as<unsigned>(),
        conf["hidden_dim"].as<unsigned>(),
        corpus.n_labels,
        pretrained,
        corpus.id_to_label);
  } else {
    _ERROR << "Unknown graph type:" << conf["graph"].as<std::string>();
    exit(1);
  }

  corpus.load_devel_data(conf["dev_data"].as<std::string>());
  _INFO << "loaded " << corpus.n_devel << " devel sentences.";
#ifdef _MSC_VER
  std::string tmp_output = "lstm.ner.evaluator";
#else
  std::string tmp_output = "/tmp/lstm.ner.evaluator." + boost::lexical_cast<std::string>(getpid());
#endif
  if (conf.count("train")) {
    train(conf, model, *engine, model_name, tmp_output, training_vocab, singletons);
  }

  std::ifstream in(model_name);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  evaluate(conf, *engine, tmp_output, training_vocab);
  engine->explain(std::cout, corpus.id_to_word, corpus.id_to_postag, corpus.id_to_label);
  return 0;
}
