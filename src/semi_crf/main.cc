#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <ctime>
#include "cnn/cnn.h"
#include "cnn/model.h"
#include "cnn/expr.h"
#include "cnn/training.h"
#include "logging.h"
#include "utils.h"
#include "training_utils.h"
#include "semi_crf/semi_crf.h"
#include "semi_crf/corpus.h"
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>


namespace po = boost::program_options;
Corpus corpus;

void init_command_line(int argc, char* argv[], po::variables_map* conf) {
  po::options_description desc("Semi-CRF lstm parser");
  desc.add_options()
    ("train,t", "Use to specify perform training")
    ("graph", po::value<std::string>()->default_value("rnn"),
    "The graph for seg-nn [rnn,cnn,concate,rnn_lex,concate_lex,rnn_rich_lex,concate_rich_lex,rnn_tuned_rich_lex,concate_tuned_rich_lex]")
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), "The optimizer.")
    ("loss", po::value<std::string>()->default_value("crossentropy"), "The loss type [crossentropy, margin].")
    ("training_data,T", po::value<std::string>(), "The path to the training data.")
    ("devel_data,d", po::value<std::string>(), "The path to the development data.")
    ("pretrained,w", po::value<std::string>(), "The path to the pretrained word embedding.")
    ("lexicon", po::value<std::string>(), "The path to the lexicon.")
    ("model,m", po::value<std::string>(), "The path to the model file, used during testing.")
    ("unk_strategy,o", po::value<unsigned>()->default_value(1), "Unknown word strategy.")
    ("unk_prob,u", po::value<double>()->default_value(0.2), "Probabilty with which to replace singletons.")
    ("layers", po::value<unsigned>()->default_value(1), "number of layers for LSTM")
    ("word_dim", po::value<unsigned>()->default_value(32), "input embedding size")
    ("pos_dim", po::value<unsigned>()->default_value(12), "POS dimension")
    ("pretrained_dim", po::value<unsigned>()->default_value(50), "pretrained input dimension")
    ("label_dim", po::value<unsigned>()->default_value(16), "label embedding size")
    ("lex_dim", po::value<unsigned>()->default_value(8), "The dimension for lexicon.")
    ("lstm_input_dim", po::value<unsigned>()->default_value(60), "The dimension for lstm")
    ("seg_dim", po::value<unsigned>()->default_value(48), "The dimension for segment.")
    ("filters", po::value<std::string>()->default_value("2*1"), "The filters: in format of <width1>*<nb2>:<width2>*<nb2>:...")
    ("hidden1_dim", po::value<unsigned>()->default_value(64), "The dimension for the first hidden layer.")
    ("hidden2_dim", po::value<unsigned>()->default_value(48), "The dimension for the second hidden layer.")
    ("duration_dim", po::value<unsigned>()->default_value(4), "The dimension for duration embedding.")
    ("max_seg_len", po::value<unsigned>()->default_value(10), "The max segmentation length.")
    ("init_segment_embedding", "Initialize the segment embedding with pretrained")
    ("eta", po::value<float>(), "the learning rate")
    ("dropout", po::value<float>()->default_value(0.), "the rate for dropout")
    ("maxiter", po::value<unsigned>()->default_value(10), "The number of max iteration")
    ("output", po::value<std::string>(), "the path to output file.")
    ("conlleval", po::value<std::string>()->default_value("./conlleval.sh"), "config path to the conlleval script")
    ("report_stops", po::value<unsigned>()->default_value(100), "the number of stops for reporting.")
    ("evaluate_stops", po::value<unsigned>()->default_value(2500), "the number of stops for evaluation.")
    ("verbose,v", "verbose log")
    ("help,h", "show a list of help information");

  po::store(po::parse_command_line(argc, argv, desc), *conf);

  if (conf->count("help")) {
    std::cerr << desc << std::endl;
    exit(1);
  }
  init_boost_log(conf->count("verbose"));
  if (!conf->count("training_data")) {
    std::cerr << "Please specify --training_data (-T), even in test" << std::endl;
    exit(1);
  }
}

std::string get_model_name(const po::variables_map& conf) {
  std::ostringstream os;
  os << "semicrf" << "_" << conf["layers"].as<unsigned>()
    << "_" << conf["word_dim"].as<unsigned>()
    << "_" << conf["pos_dim"].as<unsigned>()
    << "_" << conf["label_dim"].as<unsigned>()
    << "_" << conf["lstm_input_dim"].as<unsigned>()
    << "_" << conf["hidden1_dim"].as<unsigned>()
    << "_" << conf["hidden2_dim"].as<unsigned>()
    << "_" << conf["seg_dim"].as<unsigned>()
    << "_" << conf["duration_dim"].as<unsigned>()
#ifndef _MSC_VER
    << "_" << getpid()
#endif
    << ".params";
  return os.str();
}


bool check_max_seg(const Corpus::Segmentation& segment,
  unsigned max_seg_len) {
  for (auto& token : segment) {
    if (token.second > max_seg_len) {
      _WARN << "skip: max_seg_len=" << max_seg_len << " but reference is " << token.second;
      return false; 
    }
  }
  return true;
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

void segmentation_to_tags(const Corpus::Segmentation& segmentation, std::vector<std::string>& output) {
  output.clear();
  for (unsigned i = 0; i < segmentation.size(); ++i) {
    std::string tag = corpus.id_to_label[segmentation[i].first];
    if (tag == "O") {
      for (unsigned j = 0; j < segmentation[i].second; ++j) { output.push_back("O"); }
    } else {
      if (segmentation[i].second == 1) { output.push_back("S-" + tag); }
      else {
        for (unsigned j = 0; j < segmentation[i].second; ++j) {
          if (j == 0) { output.push_back("B-" + tag); }
          else if (j == segmentation[i].second - 1) { output.push_back("E-" + tag); }
          else { output.push_back("I-" + tag); }
        }
      }
    }
  }
}

double evaluate(const po::variables_map& conf,
  SemiCRFBuilderI& crf,
  const std::string& tmp_output,
  const std::set<unsigned>& training_vocab) {
  auto kUNK = corpus.get_or_add_word(Corpus::UNK);
  auto t_start = std::chrono::high_resolution_clock::now();

  std::ofstream ofs(tmp_output);
  unsigned max_seg_len = conf["max_seg_len"].as<unsigned>();
  for (unsigned sid = 0; sid < corpus.n_devel; ++sid) {
    const Corpus::Sentence& raw_sentence = corpus.devel_sentences[sid];
    const Corpus::Sentence& postag = corpus.devel_postags[sid];
    const Corpus::RawSentence& sentence_str = corpus.devel_sentences_str[sid];
    const Corpus::Segmentation& segmentation = corpus.devel_segmentations[sid];
    
    unsigned len = sentence_str.size();
    Corpus::Sentence sentence = raw_sentence;
    for (auto& w : sentence) {
      if (!training_vocab.count(w)) { w = kUNK; }
    }
    
    cnn::ComputationGraph cg;
    Corpus::Segmentation yz_pred;
    crf.viterbi(cg, raw_sentence, sentence, postag, yz_pred, max_seg_len);
    std::vector<std::string> predict_tags;
    std::vector<std::string> correct_tags;
    segmentation_to_tags(segmentation, correct_tags);
    segmentation_to_tags(yz_pred, predict_tags);
    BOOST_ASSERT_MSG(correct_tags.size() == len, "Number of correct tags not equal to sentence len.");
    BOOST_ASSERT_MSG(predict_tags.size() == len, "Number of predict tags not equal to sentence len.");

    for (unsigned i = 0; i < len; ++i) {
      ofs << sentence_str[i] << " " << correct_tags[i] << " " << predict_tags[i] << std::endl;
    }
    ofs << std::endl;
  }
  ofs.close();
  auto t_end = std::chrono::high_resolution_clock::now();
  double f_score = conlleval(conf, tmp_output);
  _INFO << "Test f-score: " << f_score << " [" << corpus.n_devel <<
    " sents in " << std::chrono::duration<double, std::milli>(t_end - t_start).count() << " ms]";
  return f_score;
}

void train(const po::variables_map& conf,
  cnn::Model& model,
  SemiCRFBuilderI& crf,
  const std::string& model_name,
  const std::string& tmp_output,
  const std::set<unsigned>& vocabulary,
  const std::set<unsigned>& singletons) {
  _INFO << "start semi-crf training ...";

  // Setup the trainer
  cnn::Trainer* trainer = get_trainer(conf, &model);

  // Order for shuffle
  std::vector<unsigned> order(corpus.n_train);
  for (unsigned i = 0; i < corpus.n_train; ++i) { order[i] = i; }

  unsigned kUNK = corpus.get_or_add_word(Corpus::UNK);
  auto maxiter = conf["maxiter"].as<unsigned>();
  unsigned max_seg_len = conf["max_seg_len"].as<unsigned>();

  double n_seen = 0;
  float llh = 0.f; float n_tags = 0.f;
  float llh_in_batch = 0.f; float n_tags_in_batch = 0.f;

  double best_f = 0.f;
  unsigned logc = 0;

  auto unk_strategy = conf["unk_strategy"].as<unsigned>();
  auto unk_prob = conf["unk_prob"].as<double>();

  for (unsigned iter = 0; iter < maxiter; ++iter) {
    _INFO << "start training iteration #" << iter << ", traininig data shuffled";
    std::shuffle(order.begin(), order.end(), (*cnn::rndeng));

    for (unsigned i = 0; i < order.size(); ++i) {
      unsigned sid = order[i];
      const Corpus::Sentence& raw_sentence = corpus.train_sentences[sid];
      const Corpus::Sentence& postag = corpus.train_postags[sid];
      const Corpus::Segmentation& segmentation = corpus.train_segmentations[sid];

      Corpus::Sentence sentence = raw_sentence;
      if (unk_strategy == 1) {
        for (auto& w : sentence) {
          if (singletons.count(w) && cnn::rand01() < unk_prob) { w = kUNK; }
        }
      }

      double lp;
      if (check_max_seg(segmentation, max_seg_len)) {
        cnn::ComputationGraph cg;
        if (conf["loss"].as<std::string>() == "crossentropy") {
          crf.supervised_loss(cg, raw_sentence,sentence, postag, segmentation, max_seg_len);
        } else if (conf["loss"].as<std::string>() == "margin") {
          crf.margin_loss(cg, raw_sentence, sentence, postag, segmentation, max_seg_len);
        }
        n_tags_in_batch += segmentation.size();
        lp = cnn::as_scalar(cg.forward());
        BOOST_ASSERT_MSG(lp >= 0, "Log prob < 0 on sentence");
        cg.backward();
        trainer->update(1.f);
        llh += lp; llh_in_batch += lp;
        n_tags += segmentation.size(); n_tags_in_batch += segmentation.size();
      } 

      n_seen += 1;
      ++logc;
      if (logc % conf["report_stops"].as<unsigned>() == 0) {
        trainer->status();
        _INFO << "iter (batch) #" << iter << " (epoch " << n_seen / corpus.n_train
          << ") llh: " << llh_in_batch << " ppl: " << exp(llh_in_batch / n_tags_in_batch);
        llh += llh_in_batch;
        llh_in_batch = n_tags_in_batch = 0.f;
      }
      if (logc % conf["evaluate_stops"].as<unsigned>() == 0) {
        double f = evaluate(conf, crf, tmp_output, vocabulary);
        if (f > best_f) {
          best_f = f;
          _INFO << "new best record achieved: " << best_f << " model updated.";
          std::ofstream out(model_name);
          boost::archive::text_oarchive oa(out);
          oa << model;
        }
      }
    }

    _INFO << "iter (batch) #" << iter << " (epoch " << n_seen / corpus.n_train
      << ") llh: " << llh << " ppl: " << exp(llh / n_tags);
    llh = n_tags = 0.f;

    double f = evaluate(conf, crf, tmp_output, vocabulary);
    if (f > best_f) {
      best_f = f;
      _INFO << "new best record achieved: " << best_f << ", model updated.";
      std::ofstream out(model_name);
      boost::archive::text_oarchive oa(out);
      oa << model;
    }

    if (conf["optimizer"].as<std::string>() == "simple_sgd" || conf["optimizer"].as<std::string>() == "momentum_sgd") {
      trainer->update_epoch();
    }
  }
}

void load_lexicon(const std::string& filename, std::unordered_map<HashVector, unsigned>& lexicon) {
  std::string line;
  std::ifstream ifs(filename);

  while (std::getline(ifs, line)) {
    boost::algorithm::trim(line);
    std::size_t p = line.rfind('\t');
    BOOST_ASSERT_MSG(p != std::string::npos, "delimiter not found");
    std::vector<std::string> tokens;
    HashVector key;
    std::string buf = line.substr(0, p);
    boost::algorithm::split(tokens, buf, boost::is_any_of(" "), boost::token_compress_on);
    bool all_found = true;
    for (auto& token : tokens) {
      unsigned wid;
      if (!corpus.find(token, corpus.word_to_id, wid)) { all_found = false; break; }
      key.push_back(wid);
    }
    if (all_found) {
      lexicon[key] = boost::lexical_cast<unsigned>(line.substr(p + 1));
    }
  }
  _INFO << "loaded " << lexicon.size() << " entries.";
}

void load_rich_lexicon(const std::string& filename, 
  std::unordered_map<HashVector, unsigned>& lexicon,
  std::vector<std::vector<float>>& embedding) {
  std::string line;
  std::ifstream ifs(filename);

  unsigned id = 1;
  lexicon[HashVector()] = 0;
  embedding.push_back(std::vector<float>()); // placeholder

  while (std::getline(ifs, line)) {
    boost::algorithm::trim(line);
    std::size_t p = line.rfind('\t');
    BOOST_ASSERT_MSG(p != std::string::npos, "delimiter not found");
    std::vector<std::string> tokens;
    HashVector key;
    std::string buf = line.substr(0, p);
    boost::algorithm::split(tokens, buf, boost::is_any_of(" "), boost::token_compress_on);
    bool all_found = true;
    for (auto& token : tokens) {
      unsigned wid;
      if (!corpus.find(token, corpus.word_to_id, wid)) { all_found = false; break; }
      key.push_back(wid);
    }
    if (all_found) {
      std::vector<float> values;
      buf = line.substr(p + 1);
      boost::algorithm::split(tokens, buf, boost::is_any_of(" "), boost::token_compress_on);
      values.resize(tokens.size());
      for (unsigned i = 0; i < values.size(); ++i) { values[i] = boost::lexical_cast<float>(tokens[i]); }
      lexicon[key] = id; id += 1;
      embedding.push_back(values);
    }
  }
  embedding[0].resize(embedding[1].size(), 0.);
  _INFO << "loaded " << lexicon.size() << " entries with " << embedding[0].size() << " dim";
}

int main(int argc, char* argv[]) {
  cnn::Initialize(argc, argv, 1234);
  std::cerr << "command:";
  for (int i = 0; i < argc; ++i) std::cerr << ' ' << argv[i];
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
  _INFO << "vocabulary size (after loaded training data): " << corpus.word_to_id.size();

  std::set<unsigned> training_vocab, singleton;
  corpus.get_vocabulary_and_singletons(training_vocab, singleton);

  std::unordered_map<unsigned, std::vector<float>> pretrained;
  if (conf.count("pretrained")) {
    // TODO: currently only use pretrained word embedding to initialize the training data.
    load_pretrained_word_embedding(conf["pretrained"].as<std::string>(),
      conf["pretrained_dim"].as<unsigned>(), pretrained, corpus);
    _INFO << "pretrained word embedding is loaded.";
  }

  _INFO << "vocabulary size (after loaded pretrained word embedding): " << corpus.word_to_id.size();

  // Two lexicons
  std::unordered_map<HashVector, unsigned> lexicon;
  std::vector<std::vector<float>> entity_embeddings;
  const std::string graph = conf["graph"].as<std::string>();
  if (conf.count("lexicon")) {
    if (boost::algorithm::ends_with(graph, "_rich_lex")) {
      load_rich_lexicon(conf["lexicon"].as<std::string>(), lexicon, entity_embeddings);
    } else if (boost::algorithm::ends_with(graph, "_lex")) {
      load_lexicon(conf["lexicon"].as<std::string>(), lexicon);
    }
  }

  cnn::Model model;
  SemiCRFBuilderI* crf = nullptr;

  if (graph == "rnn") {
    crf = new RNNSemiCRFBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      pretrained);
  } else if (graph == "cnn") {
    std::vector<std::pair<unsigned, unsigned>> filters;
    std::vector<std::string> tokens;
    boost::algorithm::split(tokens, conf["filters"].as<std::string>(), boost::is_any_of(":"));
    unsigned dim_seg = 0;
    for (auto& token : tokens) {
      std::size_t p = token.find('*');
      BOOST_ASSERT_MSG(p != std::string::npos, "Unexpected filter config format.");
      std::string width_str = token.substr(0, p);
      std::string nb_str = token.substr(p + 1);
      filters.push_back(std::make_pair(boost::lexical_cast<unsigned>(width_str), 
        boost::lexical_cast<unsigned>(nb_str)));
      _INFO << "Filter: " << token;
      dim_seg += conf["hidden1_dim"].as<unsigned>();
    }
    crf = new CNNSemiCRFBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      filters,
      dim_seg,
      conf["duration_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      pretrained);
  } else if (graph == "concate") {
    crf = new ConcateSemiCRFBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["max_seg_len"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      pretrained);
  } else if (graph == "simple_concate") {
    crf = new SimpleConcateSemiCRFBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["max_seg_len"].as<unsigned>(),
      conf["max_seg_len"].as<unsigned>() * conf["hidden1_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      pretrained);
  } else if (graph == "rnn_lex") {
    crf = new RNNSemiCRFwLexBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      (1 << corpus.id_to_label.size()), conf["lex_dim"].as<unsigned>(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      lexicon,
      pretrained);
  } else if (graph == "concate_lex") {
    crf = new ConcateSemiCRFwLexBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      (1 << corpus.id_to_label.size()), conf["lex_dim"].as<unsigned>(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["max_seg_len"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      lexicon,
      pretrained);
  } else if (graph == "rnn_rich_lex") {
    crf = new RNNSemiCRFwRichLexBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      lexicon.size() + 10, entity_embeddings[1].size(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      lexicon,
      entity_embeddings,
      pretrained);
  } else if (graph == "concate_rich_lex") {
    crf = new ConcateSemiCRFwRichLexBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      lexicon.size() + 10, entity_embeddings[1].size(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["max_seg_len"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      lexicon,
      entity_embeddings,
      pretrained);
  } else if (graph == "rnn_tuned_rich_lex") {
    crf = new RNNSemiCRFwTuneRichLexBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      lexicon.size() + 10, entity_embeddings[1].size(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      lexicon,
      entity_embeddings,
      pretrained,
      conf.count("init_segment_embedding"));
  } else if (graph == "concate_tuned_rich_lex") {
    crf = new ConcateSemiCRFwTuneRichLexBuilder(model,
      training_vocab.size() + 10, conf["word_dim"].as<unsigned>(),
      corpus.max_postag + 10, conf["pos_dim"].as<unsigned>(),
      corpus.max_word + 1, conf["pretrained_dim"].as<unsigned>(),
      corpus.id_to_label.size(), conf["label_dim"].as<unsigned>(),
      lexicon.size() + 10, entity_embeddings[1].size(),
      conf["layers"].as<unsigned>(),
      conf["lstm_input_dim"].as<unsigned>(),
      conf["hidden1_dim"].as<unsigned>(),
      conf["max_seg_len"].as<unsigned>(),
      conf["seg_dim"].as<unsigned>(),
      conf["duration_dim"].as<unsigned>(),
      conf["hidden2_dim"].as<unsigned>(),
      conf["dropout"].as<float>(),
      lexicon,
      entity_embeddings,
      pretrained,
      conf.count("init_segment_embedding"));
  } else {
    _ERROR << "unknown graph: " << graph;
    abort();
  }

  corpus.load_devel_data(conf["devel_data"].as<std::string>());
  _INFO << "vocabulary size (after loaded development data): " << corpus.word_to_id.size();

  std::string tmp_output;
  if (conf.count("output")) {
    tmp_output = conf["output"].as<std::string>();
  } else {
#ifdef _MSC_VER
    tmp_output = "semi_crf.evaluator";
#else
    tmp_output = "/tmp/semi_crf.evaluator." + boost::lexical_cast<std::string>(getpid());
#endif
  }

  if (conf.count("train") && conf.count("model")) {
    std::ifstream fin(conf["model"].as<std::string>());
    boost::archive::text_iarchive ia(fin);
    ia >> model;
  }

  if (conf.count("train")) {
    train(conf, model, *crf, model_name, tmp_output, training_vocab, singleton);
  }

  std::ifstream fin(model_name);
  boost::archive::text_iarchive ia(fin);
  ia >> model;
  evaluate(conf, *crf, tmp_output, training_vocab);
  return 0;
}
