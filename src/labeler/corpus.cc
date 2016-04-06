#include "utils.h"
#include "logging.h"
#include "corpus.h"
#include <boost/assert.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>


Corpus::Corpus()
  : max_word(0), max_postag(0), max_char(0) {
}


void Corpus::load_training_data(const std::string& filename) {
  std::ifstream train_file(filename);
  BOOST_ASSERT_MSG(train_file, "Failed to open training file.");
  std::string line;

  word_to_id[Corpus::BAD0] = 0; id_to_word[0] = Corpus::BAD0;
  word_to_id[Corpus::UNK] = 1;  id_to_word[1] = Corpus::UNK;

  BOOST_ASSERT_MSG(max_word == 0, "max_word is set before loading training data!");
  max_word = 2;

  postag_to_id[Corpus::BAD0] = 0; id_to_postag[0] = Corpus::BAD0;
  BOOST_ASSERT_MSG(max_postag == 0, "max_postag is set before loading training data!");
  max_postag = 1;

  char_to_id[Corpus::BAD0] = 1; id_to_char[1] = Corpus::BAD0;
  max_char = 1;

  std::vector<unsigned> current_sentence;
  std::vector<unsigned> current_postag;
  std::vector<unsigned> current_label;

  unsigned sid = 0;
  while (std::getline(train_file, line)) {
    replace_string_in_place(line, "-RRB-", "_RRB_");
    replace_string_in_place(line, "-LRB-", "_LRB_");

    if (line.empty()) {
      if (current_sentence.size() == 0) {
        // To handle the leading empty line.
        continue;
      }
      train_sentences[sid] = current_sentence;
      train_postags[sid] = current_postag;
      train_labels[sid] = current_label;

      sid ++;
      n_train = sid;
      current_sentence.clear();
      current_postag.clear();
      current_label.clear();
    } else {
      boost::algorithm::trim(line);
      std::vector<std::string> items;
      boost::algorithm::split(items, line, boost::is_any_of("\t "), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() == 4, "Ill formated CoNLL data");
      const std::string& word = items[0];
      const std::string& pos = items[1];
      const std::string& label = items[3];

      add(pos, max_postag, postag_to_id, id_to_postag);
      add(word, max_word, word_to_id, id_to_word);

      unsigned lid;
      bool found = find(label, id_to_label, lid);

      if (!found) {
        id_to_label.push_back(label);
        lid = id_to_label.size() - 1;
      }

      current_sentence.push_back(word_to_id[word]);
      current_postag.push_back(postag_to_id[pos]);
      current_label.push_back(lid);
    }
  }

  if (current_sentence.size() > 0) {
    train_sentences[sid] = current_sentence;
    train_postags[sid] = current_postag;
    train_labels[sid] = current_label;
    n_train = sid + 1;
  }

  train_file.close();
  _INFO << "finish loading training data.";
  
  n_labels = id_to_label.size();
  stat();
}


void Corpus::load_devel_data(const std::string& filename) {
  std::ifstream devel_file(filename);
  std::string line;

  BOOST_ASSERT_MSG(max_word > 3, "max_word is not set before loading development data!");
  BOOST_ASSERT_MSG(max_postag > 1, "max_postag is not set before loading development data!");

  std::vector<unsigned> current_sentence;
  std::vector<std::string> current_sentence_str;
  std::vector<unsigned> current_postag;
  std::vector<unsigned> current_label;

  unsigned sid = 0;
  while (std::getline(devel_file, line)) {
    replace_string_in_place(line, "-RRB-", "_RRB_");
    replace_string_in_place(line, "-LRB-", "_LRB_");

    if (line.empty()) {
      if (current_sentence.size() == 0) {
        // To handle the leading empty line.
        continue;
      }
      devel_sentences[sid] = current_sentence;
      devel_sentences_str[sid] = current_sentence_str;
      devel_postags[sid] = current_postag;
      devel_labels[sid] = current_label;

      sid++;
      n_devel = sid;
      current_sentence.clear();
      current_sentence_str.clear();
      current_postag.clear();
      current_label.clear();
    } else {
      boost::algorithm::trim(line);
      std::vector<std::string> items;
      boost::algorithm::split(items, line, boost::is_any_of("\t "), boost::token_compress_on);

      BOOST_ASSERT_MSG(items.size() == 4, "Ill formated CoNLL data");
      const std::string& word = items[0];
      const std::string& pos = items[1];
      const std::string& label = items[3];

      current_sentence_str.push_back(word);

      unsigned payload = 0;
      if (!find(pos, postag_to_id, payload)) {
        BOOST_ASSERT_MSG(false, "Unknow postag in development data.");
      } else {
        current_postag.push_back(payload);
      }

      if (!find(word, word_to_id, payload)) {
        find(Corpus::UNK, word_to_id, payload);
        current_sentence.push_back(payload);
      } else {
        current_sentence.push_back(payload);
      }
      
      bool found = find(label, id_to_label, payload);
      if (!found) {
        BOOST_ASSERT_MSG(false, "Unknow label in development data.");
      }

      current_label.push_back(payload);
    }
  }

  if (current_sentence.size() > 0) {
    devel_sentences[sid] = current_sentence;
    devel_sentences_str[sid] = current_sentence_str;
    devel_postags[sid] = current_postag;
    devel_labels[sid] = current_label;
    n_devel = sid + 1;
  }

  devel_file.close();
  _INFO << "finish load development data.";
}


void Corpus::stat() const {
  _INFO << "action indexing ...";
  for (auto l : id_to_label) {
    _INFO << l;
  }
  _INFO << "number of labels: " << id_to_label.size();
  _INFO << "max id of words: " << max_word;
  _INFO << "max id of postags: " << max_postag;

  _INFO << "postag indexing ...";
  for (auto& p : id_to_postag) { _INFO << p.first << " : " << p.second; }
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
