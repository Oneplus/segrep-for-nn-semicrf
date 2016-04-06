#ifndef __CPYPDICT_H__
#define __CPYPDICT_H__

#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>
#include <vector>
#include <functional>
#include "utils.h"

struct Corpus: public CorpusI {
  std::map<int, std::vector<unsigned>> train_labels;
  std::map<int, std::vector<unsigned>> train_sentences;
  std::map<int, std::vector<unsigned>> train_postags;

  std::map<int, std::vector<unsigned>> devel_labels;
  std::map<int, std::vector<unsigned>> devel_sentences;
  std::map<int, std::vector<unsigned>> devel_postags;
  std::map<int, std::vector<std::string>> devel_sentences_str;

  unsigned n_train; /* number of sentences in the training data */
  unsigned n_devel; /* number of sentences in the development data */

  unsigned n_labels;

  unsigned max_word;
  StringToIdMap word_to_id;
  IdToStringMap id_to_word;

  unsigned max_postag;
  StringToIdMap postag_to_id;
  IdToStringMap id_to_postag;

  unsigned max_char;
  StringToIdMap char_to_id;
  IdToStringMap id_to_char;

  std::vector<std::string> id_to_label;

  Corpus();
  
  void load_training_data(const std::string& filename);
  void load_devel_data(const std::string& filename);
  void stat() const;
  unsigned get_or_add_word(const std::string& word);
  void get_vocabulary_and_singletons(std::set<unsigned>& vocabulary, std::set<unsigned>& singletons);
};

#endif  //  end for __CPYPDICT_H__
