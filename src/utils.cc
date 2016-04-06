#include <iostream>
#include <fstream>
#include <sstream>
#include "utils.h"
#include "logging.h"

#if _MSC_VER
const char* CorpusI::UNK = "UNK";
const char* CorpusI::BAD0 = "<BAD0>";
#endif


unsigned UTF8len(unsigned char x) {
  if (x < 0x80) {
    return 1;
  } else if ((x >> 5) == 0x06) {
    return 2;
  } else if ((x >> 4) == 0x0e) {
    return 3;
  } else if ((x >> 3) == 0x1e) {
    return 4;
  } else if ((x >> 2) == 0x3e) {
    return 5; 
  } else if ((x >> 1) == 0x7e) {
    return 6;
  } else {
    return 0;
  }
}


void replace_string_in_place(std::string& subject,
  const std::string& search,
  const std::string& replace) {
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}


bool CorpusI::add(const std::string& str, unsigned& max_id,
  CorpusI::StringToIdMap& str_to_id, CorpusI::IdToStringMap& id_to_str) {
  if (str_to_id.find(str) == str_to_id.end()) {
    str_to_id[str] = max_id;
    id_to_str[max_id] = str;
    ++max_id;
    return true;
  }
  return false;
}


bool CorpusI::find(const std::string& str, const std::vector<std::string>& id_to_label,
  unsigned& id) const {
  for (unsigned i = 0; i < id_to_label.size(); ++i) {
    auto l = id_to_label[i];
    if (l == str) {
      id = i;
      return true;
    }
  }
  id = id_to_label.size();
  return false;
}


bool CorpusI::find(const std::string& str, const CorpusI::StringToIdMap& str_to_id,
  unsigned& id) const {
  auto result = str_to_id.find(str);
  if (result != str_to_id.end()) {
    id = result->second;
    return true;
  }
  id = 0;
  return false;
}


void load_pretrained_word_embedding(const std::string& embedding_file,
  unsigned pretrained_dim,
  std::unordered_map<unsigned, std::vector<float> >& pretrained,
  CorpusI& corpus) {
  // Loading the pretrained word embedding from file. Embedding file is presented
  // line by line, with the lexical word in leading place and 50 real number follows.  
  pretrained[corpus.get_or_add_word(CorpusI::UNK)] = std::vector<float>(pretrained_dim, 0);
  _INFO << "Loading from " << embedding_file << " with " << pretrained_dim << " dimensions";
  std::ifstream in(embedding_file);
  std::string line;
  std::getline(in, line);
  std::vector<float> v(pretrained_dim, 0);
  std::string word;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    iss >> word;
    for (unsigned i = 0; i < pretrained_dim; ++i) { iss >> v[i]; }
    unsigned id = corpus.get_or_add_word(word);
    pretrained[id] = v;
  }
}

double segmentation_loss(const std::vector<std::pair<unsigned, unsigned>>& yz_gold,
  const std::vector<std::pair<unsigned, unsigned>>& yz_pred) {
  std::set<std::tuple<unsigned, unsigned, unsigned>> gold;
  std::set<std::tuple<unsigned, unsigned, unsigned>> pred;

  unsigned cur = 0;
  for (auto& yz : yz_gold) {
    unsigned j = cur + yz.second;
    gold.insert(std::make_tuple(cur, j, yz.first));
    cur = j;
  }

  cur = 0;
  for (auto& yz : yz_pred) {
    unsigned j = cur + yz.second;
    pred.insert(std::make_tuple(cur, j, yz.first));
    cur = j;
  }

  double n_corr = 0.;
  for (auto& g : gold) {
    if (pred.count(g)) { n_corr++; }
  }
  return 0.2 * (double(gold.size()) + double(pred.size()) - 2 * n_corr);
}

