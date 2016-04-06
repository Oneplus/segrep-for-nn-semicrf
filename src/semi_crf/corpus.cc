#include "logging.h"
#include "corpus.h"
#include <fstream>
#include <sstream>
#include <cstdio>
#include <boost/assert.hpp>
#include <boost/algorithm/string.hpp>


Corpus::Corpus(): n_train(0), n_devel(0), max_word(0), max_postag(0) {

}


void Corpus::load_training_data(const std::string& filename) {
  _INFO << "Reading training data from " << filename << " ...";
  add(Corpus::BAD0, max_word, word_to_id, id_to_word);
  add(Corpus::UNK, max_word, word_to_id, id_to_word);

  std::ifstream in(filename);
  BOOST_ASSERT(in);
  std::string line;

  unsigned sid = 0;
  RawSentence dummy;
  while (std::getline(in, line)) {
    boost::algorithm::trim(line);
    if (line.size() == 0) { continue; }
    parse_line(line, train_sentences[sid], dummy, train_postags[sid],
      train_segmentations[sid], true);
    ++sid;
  }
  n_train = train_sentences.size();
  _INFO << "action indexing ...";
  for (auto l : id_to_label) { _INFO << " - " << l; }
  _INFO << "postag indexing ...";
  for (auto& p : id_to_postag) { _INFO << p.first << " : " << p.second; }
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
      devel_postags[sid], devel_segmentations[sid], false);
    ++sid;
  }
  n_devel = devel_sentences.size();
}


void Corpus::parse_line(const std::string& line, 
  Corpus::Sentence& sentence,
  Corpus::RawSentence& raw_sentence,
  Corpus::Sentence& postag,
  Corpus::Segmentation& segmentation, 
  bool train) {

  std::istringstream in(line);
  std::string token;
  std::vector<std::pair<int, unsigned>> yz;
  while (true) {
    in >> token; // word_postag
    if (token == "|||") break;
    size_t p = token.rfind("_");
    if (p == std::string::npos || p == 0 || p == (token.size() - 1)) {
      BOOST_ASSERT_MSG(false, "Ill formatted data.");
    }
    std::string word = token.substr(0, p);
    std::string pos = token.substr(p + 1);
    if (train) {
      add(word, max_word, word_to_id, id_to_word);
      add(pos, max_postag, postag_to_id, id_to_postag);
      sentence.push_back(word_to_id[word]);
      postag.push_back(postag_to_id[pos]);
    } else {
      unsigned payload = 0;
      if (!find(pos, postag_to_id, payload)) {
        BOOST_ASSERT_MSG(false, "Unknown postag in development data.");
      } else {
        postag.push_back(postag_to_id[pos]);
      }

      if (!find(word, word_to_id, payload)) {
        find(Corpus::UNK, word_to_id, payload);
      }
      sentence.push_back(payload);
      raw_sentence.push_back(word);
    }
  }

  while (true) {
    in >> token;
    if (!in) break;
    unsigned p = token.rfind(':');
    if (p == std::string::npos || p == 0 || p == (token.size() - 1)) {
      BOOST_ASSERT_MSG(false, "ill formatted segmentation");
    }

    std::string label = token.substr(0, p);
    unsigned y;
    bool found = find(label, id_to_label, y);
    if (!found) {
      if (train) {
        id_to_label.push_back(label);
        y = id_to_label.size() - 1;
      } else {
        BOOST_ASSERT_MSG(false, "Unknow label in development data.");
      }
    }

    unsigned z = atoi(token.substr(p + 1).c_str());
    segmentation.push_back(std::make_pair(y, z));
  }
}


void Corpus::stat() {
  _INFO << "# training: " << n_train;
  _INFO << "# development: " << n_devel;
}


unsigned Corpus::get_or_add_word(const std::string& word) {
  unsigned payload;
  if (!find(word, word_to_id, payload)) {
    add(word, max_word, word_to_id, id_to_word);
    return word_to_id[word];
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