#include "logging.h"
#include "corpus.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <boost/assert.hpp>
#include <boost/algorithm/string.hpp>


Corpus::Corpus() : n_train(0), n_devel(0), max_token(0) {}


void Corpus::load_training_data(const std::string& filename) {
  _INFO << "Reading training data from " << filename << " ...";
  std::ifstream in(filename);
  BOOST_ASSERT(in);
  std::string line;

  unsigned sid = 0;
  RawSentence dummy;
  add(Corpus::BAD0, max_token, token_to_id, id_to_token);
  add(Corpus::UNK, max_token, token_to_id, id_to_token);

  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) { continue; }
    parse_line(line, train_sentences[sid], dummy, train_segmentations[sid], true);
    ++sid;
  }
  n_train = train_sentences.size();
}


void Corpus::load_devel_data(const std::string& filename) {
  _INFO << "Reading development data from " << filename << " ...";
  std::ifstream in(filename);
  BOOST_ASSERT(in);
  std::string line;
  unsigned sid = 0;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) { continue; }
    parse_line(line, devel_sentences[sid], devel_sentences_str[sid], 
      devel_segmentations[sid], false);
    ++sid;
  }
  n_devel = devel_sentences.size();
}


void Corpus::parse_line(const std::string& line,
  Corpus::Sentence& sentence,
  Corpus::RawSentence& raw_sentence,
  Corpus::Segmentation& segmentation,
  bool train) {

  std::istringstream in(line);
  std::string token;
  std::vector<std::pair<int, unsigned>> yz;
  while (true) {
    in >> token; // word_postag
    if (token == "|||") break;

    if (train) {
      add(token, max_token, token_to_id, id_to_token);
      sentence.push_back(token_to_id[token]);
    } else {
      unsigned payload = 0;
      if (!find(token, token_to_id, payload)) {
        find(Corpus::UNK, token_to_id, payload);
      }
      sentence.push_back(payload);
      raw_sentence.push_back(token);
    }
  }

  while (true) {
    in >> token;
    if (!in) break;

    unsigned z = atoi(token.c_str());
    segmentation.push_back(z);
  }
}


void Corpus::stat() {
  _INFO << "# training: " << n_train;
  _INFO << "# development: " << n_devel;
}


unsigned Corpus::get_or_add_word(const std::string& word) {
  unsigned payload;
  if (!find(word, token_to_id, payload)) {
    add(word, max_token, token_to_id, id_to_token);
    return token_to_id[word];
  }
  return payload;
}


void Corpus::get_vocabulary_and_singletons(std::set<unsigned>& vocabulary,
  std::set<unsigned>& singletons) {
  std::map<unsigned, unsigned> counter;
  for (auto& payload : train_sentences) {
    for (auto& word : payload.second) {
      vocabulary.insert(word);
      ++counter[word];
    }
  }
  for (auto& payload : counter) {
    if (payload.second == 1) { singletons.insert(payload.first); }
  }
}