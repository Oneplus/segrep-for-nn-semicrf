#include "model.h"
#include <numeric>
#include <boost/assert.hpp>

NERModel::NERModel(const std::vector<std::string>& id_to_label_)
  : id_to_label(id_to_label_) {

}


void NERModel::get_possible_labels(std::vector<unsigned>& possible_labels) {
  // Get possible labels for starting point.
  for (unsigned i = 0; i < id_to_label.size(); ++i) {
#if 1
    possible_labels.push_back(i);
#else
    const std::string& str = id_to_label[i];
    if (str[0] == 'O' || str[0] == 'o' || str[0] == 'B' || str[0] == 'b') {
      possible_labels.push_back(i);
    }
#endif
  }
}


void NERModel::get_possible_labels(unsigned prev,
  std::vector<unsigned>& possible_labels) {
#if 1
  for (unsigned i = 0; i < id_to_label.size(); ++i) {
    possible_labels.push_back(i);
  }
#else
  BOOST_ASSERT_MSG(prev < id_to_label.size(), "Previous label id not in range.");
  const std::string& prev_str = id_to_label[prev];

  if (prev_str[0] == 'O') {
    for (unsigned i = 0; i < id_to_label.size(); ++i) {
      const std::string& str = id_to_label[i];
      if (str[0] == 'I' || str[0] == 'i') {
        continue;
      }
      possible_labels.push_back(i);
    }
  } else if (prev_str[0] == 'B') {
    for (unsigned i = 0; i < id_to_label.size(); ++i) {
      const std::string& str = id_to_label[i];
      if (str[0] == 'I' && str.substr(1) != prev_str.substr(1)) {
        continue;
      }
      possible_labels.push_back(i);
    }
  } else if (prev_str[0] == 'I') {
    for (unsigned i = 0; i < id_to_label.size(); ++i) {
      const std::string& str = id_to_label[i];
      if (str[0] == 'I' && str.substr(1) != prev_str.substr(1)) {
        continue;
      }
      possible_labels.push_back(i);
    }
  }
#endif
}


unsigned NERModel::get_best_scored_label(const std::vector<float>& scores,
  const std::vector<unsigned>& possible_labels) {
  float best_score = scores[possible_labels[0]];
  unsigned best_lid = 0;
  for (unsigned j = 1; j < possible_labels.size(); ++j) {
    if (best_score < scores[possible_labels[j]]) {
      best_score = scores[possible_labels[j]];
      best_lid = possible_labels[j];
    }
  }
  return best_lid;
}


LSTMNERLabeler::LSTMNERLabeler(cnn::Model* model,
  unsigned size_word,
  unsigned dim_word,
  unsigned size_postag,
  unsigned dim_postag,
  unsigned size_pretrained_word,
  unsigned dim_pretrained_word,
  unsigned dim_lstm_input,
  unsigned n_lstm_layers,
  unsigned dim_hidden,
  unsigned size_label,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
  const std::vector<std::string>& id_to_label):
  NERModel(id_to_label),
  input_layer(model, size_word, dim_word, size_postag, dim_postag, 
  size_pretrained_word, dim_pretrained_word, dim_lstm_input, pretrained_embedding),
  lstm_layer(model, n_lstm_layers, dim_lstm_input, dim_hidden),
  softmax_layer(model, dim_hidden, size_label),
  pretrained(pretrained_embedding) {
}


void LSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);

  unsigned len = sentence.size();
  std::vector<cnn::expr::Expression> exprs(len);
  for (unsigned i = 0; i < len; ++i) {
    auto wid = sentence[i];
    auto pid = postags[i];
    auto pre_wid = raw_sentence[i];
    if (!pretrained.count(pre_wid)) { pre_wid = 0; }
    exprs[i] = input_layer.add_input(hg, wid, pid, pre_wid);
  }

  lstm_layer.new_graph(hg);
  lstm_layer.add_inputs(hg, exprs);
  
  std::vector<cnn::expr::Expression> hidden(len);
  lstm_layer.get_outputs(hg, hidden);

  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, hidden[i]);
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
    hg->incremental_forward();
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


void BiLSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);
  auto len = sentence.size();

  std::vector<cnn::expr::Expression> exprs;
  get_inputs(hg, raw_sentence, sentence, sentence_str, postags, exprs);

  bi_lstm_layer.new_graph(hg);
  bi_lstm_layer.add_inputs(hg, exprs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bi_lstm_layer.get_outputs(hg, hidden1);

  std::vector<cnn::expr::Expression> merged1(len);
  for (unsigned i = 0; i < len; ++i) {
    merged1[i] = cnn::expr::rectify(
        merge2_layer.get_output(hg, hidden1[i].first, hidden1[i].second));
  }

  hg->incremental_forward();
  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, merged1[i]);
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


void GlobBiLSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);
  auto len = sentence.size();

  std::vector<cnn::expr::Expression> exprs;
  get_inputs(hg, raw_sentence, sentence, sentence_str, postags, exprs);

  bi_lstm_layer.new_graph(hg);
  bi_lstm_layer.add_inputs(hg, exprs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bi_lstm_layer.get_outputs(hg, hidden1);

  std::vector<cnn::expr::Expression> merged(len);
  for (unsigned i = 0; i < len; ++i) {
    merged[i] = cnn::expr::rectify(merge4_layer.get_output(hg,
      bi_lstm_layer.fw_lstm.back(),
      bi_lstm_layer.bw_lstm.back(),
      hidden1[i].first,
      hidden1[i].second));
  }

  hg->incremental_forward();
  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, merged[i]);
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


void LSTMBiLSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels)  {
  bool training_mode = (correct_labels.size() > 0);
  auto len = sentence.size();

  std::vector<cnn::expr::Expression> exprs(sentence.size());
  get_inputs(hg, raw_sentence, sentence, sentence_str, postags, exprs);

  bi_lstm_layer.new_graph(hg);
  bi_lstm_layer.add_inputs(hg, exprs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bi_lstm_layer.get_outputs(hg, hidden1);

  std::vector<cnn::expr::Expression> merged1(len);
  for (unsigned i = 0; i < len; ++i) {
    merged1[i] = cnn::expr::rectify(
        merge2_layer.get_output(hg, hidden1[i].first, hidden1[i].second));
  }

  lstm_layer.new_graph(hg);
  lstm_layer.add_inputs(hg, merged1);

  std::vector <cnn::expr::Expression> hidden2;
  lstm_layer.get_outputs(hg, hidden2);

  hg->incremental_forward();
  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) { get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, hidden2[i]);
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


void LSTMAttendBiLSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);
  auto len = sentence.size();

  std::vector<cnn::expr::Expression> exprs;
  get_inputs(hg, raw_sentence, sentence, sentence_str, postags, exprs);

  bi_lstm_layer.new_graph(hg);
  bi_lstm_layer.add_inputs(hg, exprs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bi_lstm_layer.get_outputs(hg, hidden1);
  
  std::vector<cnn::expr::Expression> merged1(len);
  for (unsigned i = 0; i < len; ++i) {
    merged1[i] = cnn::expr::rectify(
        merge2_layer.get_output(hg, hidden1[i].first, hidden1[i].second));
  }

  std::vector<cnn::expr::Expression> attentions(len);
  std::vector<unsigned> wids(n_windows);
  std::vector<cnn::expr::Expression> attented(n_windows);
  for (int i = 0; i < len; ++i) {
    int start = i - n_windows / 2, end = i + n_windows / 2;
    for (int k = 0, j = start; j <= end; ++j, ++k) {
      if (j < 0) {
        attented[k] = cnn::expr::parameter(*hg, bos[-j - 1]);
        wids[k] = n_words + (-j);
      } else if (j >= len) {
        attented[k] = cnn::expr::parameter(*hg, eos[j - len]);
        wids[k] = n_words + (j - len) + n_windows / 2;
      } else {
        attented[k] = merged1[j];
        wids[k] = raw_sentence[j];
      }
    }
    attentions[i] = attention_layer.get_output(hg, wids, attented);
  }

  lstm_layer.new_graph(hg);
  lstm_layer.add_inputs(hg, attentions);

  std::vector<cnn::expr::Expression> hidden2;
  lstm_layer.get_outputs(hg, hidden2);

  hg->incremental_forward();
  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, hidden2[i]);
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


void LSTMAttendBiLSTMNERLabeler::explain(std::ostream& os,
  const std::map<unsigned, std::string>& id_to_word,
  const std::map<unsigned, std::string>& id_to_postag,
  const std::vector<std::string>& id_to_label) {
  os << "[Attention Matrix]" << std::endl;

  for (unsigned i = 0; i < attention_layer.n_words; ++i) {
    auto found = id_to_word.find(i);
    os << (found == id_to_word.end() ? "**ERROR**" : found->second) << " score:";
    std::vector<float> vals(n_windows);
    for (unsigned j = 0; j < attention_layer.n_windows; ++j) {
      vals[j] = attention_layer.p_K[j]->values[i].v[0];
      os << " " << vals[j];
    }
    float max_val = (*std::max_element(vals.begin(), vals.end()));
    for (unsigned j = 0; j < attention_layer.n_windows; ++j) {
      vals[j] = std::exp(vals[j] - max_val);
    }
    os << " | prob:";
    float sum_val = std::accumulate(vals.begin(), vals.end(), 0.f);
    for (unsigned j = 0; j < attention_layer.n_windows; ++j) {
      os << " " << vals[j] / sum_val;
    }
    os << std::endl;
  }
}


void Seq2SeqLabelVerBiLSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);

  auto len = sentence.size();
  std::vector<cnn::expr::Expression> exprs;
  get_inputs(hg, raw_sentence, sentence, sentence_str, postags, exprs);

  bi_lstm_layer.new_graph(hg);
  bi_lstm_layer.add_inputs(hg, exprs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bi_lstm_layer.get_outputs(hg, hidden1);
  
  hg->incremental_forward();

  lstm_layer.new_graph(hg);
  std::vector<cnn::expr::Expression> globs;
  for (unsigned i = 0; i < lstm_layer.lstm.layers; ++i) {
    globs.push_back(cnn::expr::rectify(merge2_layer.get_output(hg,
            bi_lstm_layer.fw_lstm.c.back()[i], 
            bi_lstm_layer.bw_lstm.c.back()[i])));
  }
  for (unsigned i = 0; i < lstm_layer.lstm.layers; ++i) {
    globs.push_back(cnn::expr::rectify(merge2_layer.get_output(hg,
            bi_lstm_layer.fw_lstm.h.back()[i],
            bi_lstm_layer.bw_lstm.h.back()[i])));
  }
  lstm_layer.lstm.start_new_sequence(globs);
  
  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    if (i == 0) {
      lstm_layer.lstm.add_input(cnn::expr::parameter(*hg, p_go));
    } else {
      lstm_layer.lstm.add_input(cnn::expr::lookup(*hg, p_l, prev_lid));
    }
    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, lstm_layer.lstm.back());
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


Seq2SeqWordVerBiLSTMNERLabeler::Seq2SeqWordVerBiLSTMNERLabeler(cnn::Model* model,
  unsigned size_word,
  unsigned dim_word,
  unsigned size_postag,
  unsigned dim_postag,
  unsigned size_pretrained_word,
  unsigned dim_pretrained_word,
  unsigned dim_lstm_input,
  unsigned n_lstm_layers,
  unsigned dim_hidden,
  unsigned size_label,
  unsigned dim_label,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
  const std::vector<std::string>& id_to_label) :
  NERModel(id_to_label),
  dynamic_input_layer(model, size_word, dim_word, size_postag, dim_postag,
  size_pretrained_word, dim_pretrained_word, size_label, dim_label, dim_lstm_input,
  pretrained_embedding),
  p_w2l0(model->add_parameters({ dim_lstm_input, dim_word })),
  p_p2l0(model->add_parameters({ dim_lstm_input, dim_postag })),
  p_t2l0(model->add_parameters({ dim_lstm_input, dim_pretrained_word })),
  p_l0b(model->add_parameters({ dim_lstm_input, 1 })),
  p_guard(model->add_parameters({ dim_label, 1 })),
  bi_lstm_layer(model, n_lstm_layers, dim_lstm_input, dim_hidden),
  merge2_layer(model, dim_hidden, dim_hidden, dim_hidden),
  lstm_layer(model, n_lstm_layers, dim_hidden, dim_hidden),
  softmax_layer(model, dim_hidden, size_label),
  pretrained(pretrained_embedding) {
}


void Seq2SeqWordVerBiLSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);

  auto len = sentence.size();
  std::vector<cnn::expr::Expression> exprs(len);
  for (unsigned i = 0; i < len; ++i) {
    auto wid = sentence[i];
    auto pid = postags[i];
    auto pre_wid = raw_sentence[i];
    if (!pretrained.count(pre_wid)) {
      exprs[i] = cnn::expr::rectify(cnn::expr::affine_transform({
            cnn::expr::parameter(*hg, p_l0b),
            cnn::expr::parameter(*hg, p_w2l0), cnn::expr::lookup(*hg, dynamic_input_layer.p_w, wid),
            cnn::expr::parameter(*hg, p_p2l0), cnn::expr::lookup(*hg, dynamic_input_layer.p_p, pid)
            }));
    } else {
      exprs[i] = cnn::expr::rectify(cnn::expr::affine_transform({
            cnn::expr::parameter(*hg, p_l0b),
            cnn::expr::parameter(*hg, p_w2l0), cnn::expr::lookup(*hg, dynamic_input_layer.p_w, wid),
            cnn::expr::parameter(*hg, p_p2l0), cnn::expr::lookup(*hg, dynamic_input_layer.p_p, pid),
            cnn::expr::parameter(*hg, p_t2l0), cnn::expr::const_lookup(*hg, dynamic_input_layer.p_t, pre_wid)
            }));
    }
  }

  bi_lstm_layer.new_graph(hg);
  bi_lstm_layer.add_inputs(hg, exprs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bi_lstm_layer.get_outputs(hg, hidden1);

  lstm_layer.new_graph(hg);
  std::vector<cnn::expr::Expression> globs;
  for (unsigned i = 0; i < lstm_layer.lstm.layers; ++i) {
    globs.push_back(cnn::expr::rectify(merge2_layer.get_output(hg,
            bi_lstm_layer.fw_lstm.c.back()[i], 
            bi_lstm_layer.bw_lstm.c.back()[i])));
  }
  for (unsigned i = 0; i < lstm_layer.lstm.layers; ++i) {
    globs.push_back(cnn::expr::rectify(merge2_layer.get_output(hg,
            bi_lstm_layer.fw_lstm.h.back()[i],
            bi_lstm_layer.bw_lstm.h.back()[i])));
  }
  lstm_layer.lstm.start_new_sequence(globs);

  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  cnn::expr::Expression guard = cnn::expr::parameter(*hg, p_guard);
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    auto wid = sentence[i];
    auto pid = postags[i];
    auto pre_wid = raw_sentence[i];
    if (!pretrained.count(pre_wid)) { pre_wid = 0; }
    lstm_layer.lstm.add_input(
      i == 0 ?
      dynamic_input_layer.add_input2(hg, wid, pid, pre_wid, guard) :
      dynamic_input_layer.add_input2(hg, wid, pid, pre_wid, prev_lid));

    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, lstm_layer.lstm.back());
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


Seq2SeqAttentionVerLSTMNERLabeler::Seq2SeqAttentionVerLSTMNERLabeler(cnn::Model* model,
  unsigned size_word,
  unsigned dim_word,
  unsigned size_postag,
  unsigned dim_postag,
  unsigned size_pretrained_word,
  unsigned dim_pretrained_word,
  unsigned dim_lstm_input,
  unsigned n_lstm_layers,
  unsigned dim_hidden,
  unsigned size_label,
  unsigned dim_label,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
  const std::vector<std::string>& id_to_label) :
  NERModel(id_to_label),
  static_input_layer(model, size_word, dim_word, size_postag, dim_postag,
  size_pretrained_word, dim_pretrained_word, dim_lstm_input, pretrained_embedding),
  bi_lstm_layer(model, n_lstm_layers, dim_lstm_input, dim_hidden),
  p_l(model->add_lookup_parameters(size_label, { dim_label, 1 })),
  p_guard(model->add_parameters({ dim_label, 1 })),
  merge2_layer(model, dim_hidden, dim_hidden, dim_hidden),
  merge3_layer(model, dim_hidden, dim_hidden, dim_label, dim_hidden),
  lstm_layer(model, n_lstm_layers, dim_hidden, dim_hidden),
  softmax_layer(model, dim_hidden, size_label),
  pretrained(pretrained_embedding) {
}


void Seq2SeqAttentionVerLSTMNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);

  unsigned len = sentence.size();
  std::vector<cnn::expr::Expression> exprs(len);
  for (unsigned i = 0; i < len; ++i) {
    auto wid = sentence[i];
    auto pid = postags[i];
    auto pre_wid = raw_sentence[i];
    if (!pretrained.count(pre_wid)) { pre_wid = 0; }
    exprs[i] = static_input_layer.add_input(hg, wid, pid, pre_wid);
  }

  bi_lstm_layer.new_graph(hg);
  bi_lstm_layer.add_inputs(hg, exprs);
  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bi_lstm_layer.get_outputs(hg, hidden1);

  lstm_layer.new_graph(hg);
  std::vector<cnn::expr::Expression> globs;
  for (unsigned i = 0; i < lstm_layer.lstm.layers; ++i) {
    globs.push_back(cnn::expr::rectify(merge2_layer.get_output(hg,
      bi_lstm_layer.fw_lstm.c.back()[i],
      bi_lstm_layer.bw_lstm.c.back()[i])));
  }
  for (unsigned i = 0; i < lstm_layer.lstm.layers; ++i) {
    globs.push_back(cnn::expr::rectify(merge2_layer.get_output(hg,
      bi_lstm_layer.fw_lstm.h.back()[i],
      bi_lstm_layer.bw_lstm.h.back()[i])));
  }
  lstm_layer.lstm.start_new_sequence(globs);

  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs;
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    lstm_layer.lstm.add_input(
      merge3_layer.get_output(hg, hidden1[i].first, hidden1[i].second,
      (i == 0 ? cnn::expr::parameter(*hg, p_guard) : cnn::expr::lookup(*hg, p_l, prev_lid)))
      );

    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg, lstm_layer.lstm.back());
    hg->incremental_forward();
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}


RevLSTM2SeqNERLabeler::RevLSTM2SeqNERLabeler(cnn::Model* model,
  unsigned size_word,
  unsigned dim_word,
  unsigned size_postag,
  unsigned dim_postag,
  unsigned size_pretrained_word,
  unsigned dim_pretrained_word,
  unsigned dim_lstm_input,
  unsigned n_lstm_layers,
  unsigned dim_hidden,
  unsigned size_label,
  unsigned dim_label,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
  const std::vector<std::string>& id_to_label):
  NERModel(id_to_label),
  input_layer(model, size_word, dim_word, size_postag, dim_postag, size_pretrained_word,
  dim_pretrained_word, size_label, dim_label, dim_lstm_input, pretrained_embedding),
  p_guard(model->add_parameters({ dim_label, 1 })),
  lstm_layer(model, n_lstm_layers, dim_lstm_input, dim_hidden),
  softmax_layer(model, dim_hidden, size_label),
  pretrained(pretrained_embedding) {

}


void RevLSTM2SeqNERLabeler::log_probability(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  const std::vector<unsigned>& correct_labels,
  double &n_correct,
  std::vector<unsigned>& predict_labels) {
  bool training_mode = (correct_labels.size() > 0);

  unsigned len = sentence.size();
  std::vector<cnn::expr::Expression> exprs(len);
  for (unsigned i = 0; i < len; ++i) {
    auto wid = sentence[i];
    auto pid = postags[i];
    auto pre_wid = raw_sentence[i];
    if (!pretrained.count(pre_wid)) { pre_wid = 0; }
    exprs[i] = input_layer.add_input(hg, wid, pid, pre_wid);
  }
  
  lstm_layer.new_graph(hg);
  lstm_layer.lstm.start_new_sequence();
  for (unsigned i = 0; i < len; ++i) {
    lstm_layer.lstm.add_input(exprs[len - i - 1]);
  }

  unsigned prev_lid = 0;
  std::vector<cnn::expr::Expression> log_probs; 
  for (unsigned i = 0; i < len; ++i) {
    std::vector<unsigned> possible_labels;
    if (i == 0) {
      get_possible_labels(possible_labels);
    } else {
      get_possible_labels(prev_lid, possible_labels);
    }
    BOOST_ASSERT_MSG(possible_labels.size() > 0, "No possible labels, unexpected!");

    auto wid = sentence[i];
    auto pid = postags[i];
    auto pre_wid = raw_sentence[i];
    if (!pretrained.count(pre_wid)) { pre_wid = 0; }
    cnn::expr::Expression guard = cnn::expr::parameter(*hg, p_guard);
    lstm_layer.lstm.add_input(
      i == 0 ?
      input_layer.add_input2(hg, wid, pid, pre_wid, guard) :
      input_layer.add_input2(hg, wid, pid, pre_wid, prev_lid));
    cnn::expr::Expression softmax_scores = softmax_layer.get_output(hg,lstm_layer.lstm.back());
    std::vector<float> scores = cnn::as_vector(hg->get_value(softmax_scores));

    unsigned best_lid = get_best_scored_label(scores, possible_labels);
    unsigned lid = best_lid;
    if (training_mode) {
      lid = correct_labels[i];
      if (best_lid == correct_labels[i]) { ++n_correct; }
    }
    log_probs.push_back(cnn::expr::pick(softmax_scores, lid));
    predict_labels.push_back(lid);
    prev_lid = lid;
    hg->incremental_forward();
  }

  cnn::expr::Expression tot_neglogprob = -cnn::expr::sum(log_probs);
  assert(tot_neglogprob.pg != nullptr);
}
