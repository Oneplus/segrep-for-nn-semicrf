#ifndef __CORPUS_H__
#define __CORPUS_H__

#include <iostream>
#include <vector>
#include <map>
#include "utils.h"


struct Corpus: public CorpusI {
  typedef std::vector<std::pair<unsigned, unsigned>> Segmentation;
  
  std::map<unsigned, Sentence> train_sentences;
  std::map<unsigned, Sentence> train_postags;
  std::map<unsigned, Segmentation> train_segmentations;

  std::map<unsigned, Sentence> devel_sentences;
  std::map<unsigned, RawSentence> devel_sentences_str;
  std::map<unsigned, Sentence> devel_postags;
  std::map<unsigned, Segmentation> devel_segmentations;

  unsigned n_train;
  unsigned n_devel;

  unsigned max_word;
  StringToIdMap word_to_id;
  IdToStringMap id_to_word;

  unsigned max_postag;
  StringToIdMap postag_to_id;
  IdToStringMap id_to_postag;

  std::vector<std::string> id_to_label;

  Corpus();

  void load_training_data(const std::string& filename);
  void load_devel_data(const std::string& filename);
  void parse_line(const std::string& line, Sentence& sentence, 
    RawSentence& raw_sentence, Sentence& postags, Segmentation& segmentation, bool train);
  unsigned get_or_add_word(const std::string& word);
  void get_vocabulary_and_singletons(std::set<unsigned>& vocabulary, std::set<unsigned>& singletons);
  void stat();
};

#endif  //  end for __CORPUS_H__
