#ifndef __CORPUS_H__
#define __CORPUS_H__

#include <iostream>
#include <vector>
#include <map>
#include "utils.h"


struct Corpus : public CorpusI {
  typedef std::vector<unsigned> Segmentation;

  std::map<unsigned, Sentence> train_sentences;
  std::map<unsigned, Segmentation> train_segmentations;

  std::map<unsigned, Sentence> devel_sentences;
  std::map<unsigned, RawSentence> devel_sentences_str;
  std::map<unsigned, Segmentation> devel_segmentations;

  unsigned n_train;
  unsigned n_devel;

  unsigned max_token;
  StringToIdMap token_to_id;
  IdToStringMap id_to_token;

  Corpus();

  void load_training_data(const std::string& filename);
  void load_devel_data(const std::string& filename);
  void parse_line(const std::string& line, Sentence& sentence,
    RawSentence& raw_sentence, Segmentation& segmentation, bool train);
  unsigned get_or_add_word(const std::string& word);
  void get_vocabulary_and_singletons(std::set<unsigned>& vocabulary, std::set<unsigned>& singletons);
  void stat();
};

#endif  //  end for __CORPUS_H__
