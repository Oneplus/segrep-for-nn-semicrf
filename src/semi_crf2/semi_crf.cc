#include "semi_crf.h"
#include "logging.h"
#include <cstdio>
#include <fstream>
#include <boost/algorithm/string.hpp>


ZerothOrderSemiCRFBuilder::ZerothOrderSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  float dr,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embeddings)
  :
  input_layer(&m, size_char, dim_char, 0, 0, size_pretrained, dim_pretrained, lstm_input_dim, pretrained_embeddings),
  bilstm_layer(&m, n_layers, lstm_input_dim, dim_hidden1),
  merge2(&m, dim_hidden1, dim_hidden1, dim_hidden1),
  dur_emb(m, dim_duration),
  dropout_rate(dr),
  pretrained(pretrained_embeddings) {

}

cnn::expr::Expression ZerothOrderSemiCRFBuilder::supervised_loss(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Segmentation& correct,
  unsigned max_seg_len) {
  unsigned len = sentence.size();
  //
  std::vector<std::vector<bool>> is_ref(len, std::vector<bool>(len + 1, false));

  unsigned cur = 0;
  for (unsigned ri = 0; ri < correct.size(); ++ri) {
    // The beginning position is derivated from training data.
    BOOST_ASSERT_MSG(cur < len, "segment index greater than sentence length.");
    unsigned dur = correct[ri];
    if (max_seg_len && dur > max_seg_len) {
      _ERROR << "max_seg_len=" << max_seg_len << " but reference duration is " << dur;
      abort();
    }
    unsigned j = cur + dur;
    BOOST_ASSERT_MSG(j <= len, "End of segment is greater than the input sentence.");
    is_ref[cur][j] = true;
    cur = j;
  }
  BOOST_ASSERT_MSG(cur == len, "Senetence is not consumed by the input span.");

  dur_emb.new_graph(cg);
  bilstm_layer.new_graph(&cg);
  bilstm_layer.set_dropout(dropout_rate);

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned rwid = raw_sentence[i];
    inputs[i] = input_layer.add_input(&cg, wid, 0, rwid);
  }

  bilstm_layer.add_inputs(&cg, inputs);
  std::vector<BidirectionalLSTMLayer::Output> hiddens1;
  bilstm_layer.get_outputs(&cg, hiddens1);

  std::vector<cnn::expr::Expression> c(len);
  for (unsigned i = 0; i < len; ++i) {
    c[i] = cnn::expr::rectify(
      merge2.get_output(&cg, hiddens1[i].first, hiddens1[i].second));
  }
  construct_chart(cg, c, max_seg_len, true);

  // f is the expression of overall matrix, fr is the expression of reference.
  std::vector <cnn::expr::Expression> alpha(len + 1), ref_alpha(len + 1);
  std::vector<cnn::expr::Expression> f;
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
    for (unsigned i = i_start; i < j; ++i) {
      bool matches_ref = is_ref[i][j];
      cnn::expr::Expression p = factor_score(cg, i, j, true);
      
      if (i == 0) {
        f.push_back(p);
        if (matches_ref) { ref_alpha[j] = p; }
      } else {
        f.push_back(p + alpha[i]);
        if (matches_ref) { ref_alpha[j] = p + ref_alpha[i]; }
      }
    }
    alpha[j] = cnn::expr::logsumexp(f);
    // if (fr.size()) ref_alpha[j] = cnn::expr::logsumexp(fr);
  }
  return alpha.back() - ref_alpha.back();
}


void ZerothOrderSemiCRFBuilder::viterbi(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  Corpus::Segmentation& yz_pred,
  unsigned max_seg_len) {

  yz_pred.clear();
  unsigned len = sentence.size();

  dur_emb.new_graph(cg);
  bilstm_layer.new_graph(&cg);
  bilstm_layer.disable_dropout();

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned rwid = raw_sentence[i];
    inputs[i] = input_layer.add_input(&cg, wid, 0, rwid);
  }
  
  bilstm_layer.add_inputs(&cg, inputs);
  std::vector<BidirectionalLSTMLayer::Output> hiddens1;
  bilstm_layer.get_outputs(&cg, hiddens1);

  std::vector<cnn::expr::Expression> c(len);
  for (unsigned i = 0; i < len; ++i) {
    c[i] = cnn::expr::rectify(
      merge2.get_output(&cg, hiddens1[i].first, hiddens1[i].second));
  }
  construct_chart(cg, c, max_seg_len, false);

  std::vector<double> alpha(len + 1);
  std::vector<double> f;

  std::vector<std::pair<unsigned, unsigned>> ijt;
  std::vector<std::pair<unsigned, unsigned>> it;

  it.push_back(std::make_pair(0, 0));
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    ijt.clear();
    unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
    for (unsigned i = i_start; i < j; ++i) {
      cnn::expr::Expression p = factor_score(cg, i, j, false);
      
      double p_value = cnn::as_scalar(cg.get_value(p));
      if (i == 0) {
        f.push_back(p_value);
      } else {
        f.push_back(p_value + alpha[i]);
      }
      ijt.push_back(std::make_pair(i, j));
    }
    unsigned max_id = 0;
    auto max_val = f[0];
    for (unsigned id = 1; id < f.size(); ++id) {
      auto val = f[id];
      if (max_val < val) { max_val = val; max_id = id; }
    }
    alpha[j] = f[max_id];
    it.push_back(ijt[max_id]);
  }

  auto cur_j = len;
  while (cur_j > 0) {
    auto cur_i = std::get<0>(it[cur_j]);
    yz_pred.push_back(cur_j - cur_i);
    cur_j = cur_i;
  }
  std::reverse(yz_pred.begin(), yz_pred.end());
}

ZerothOrderRnnSemiCRFBuilder::ZerothOrderRnnSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_hidden2,
  unsigned dim_duration,
  unsigned dim_seg,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  : ZerothOrderSemiCRFBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, pretrained),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge3(&m, dim_seg, dim_seg, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1) {

}

void ZerothOrderRnnSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderRnnSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, bool train) {
  auto seg_ij = seg_emb(i, j - 1);
  auto dur = dur_emb.embed(j - i);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge3.get_output(&cg, seg_ij.first, seg_ij.second, dur)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, 
      cnn::expr::rectify(merge3.get_output(&cg, seg_ij.first, seg_ij.second, dur)));
  }
}

ZerothOrderConcateSemiCRFBuilder::ZerothOrderConcateSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  : ZerothOrderSemiCRFBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, pretrained),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge2(&m, dim_seg, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1) {

}

void ZerothOrderConcateSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, bool train) {
  auto seg_ij = seg_emb(i, j - 1);
  auto dur = dur_emb.embed(j - i);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(cnn::expr::rectify(merge2.get_output(&cg, seg_ij, dur)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, cnn::expr::rectify(merge2.get_output(&cg, seg_ij, dur)));
  }
}

ZerothOrderCnnSemiCRFBuilder::ZerothOrderCnnSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  const std::vector<std::pair<unsigned, unsigned>>& filters,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  : ZerothOrderSemiCRFBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, pretrained_embed),
  seg_emb(m, dim_hidden1, filters),
  merge2(&m, dim_seg, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1) {

}

void ZerothOrderCnnSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderCnnSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  return dense.get_output(&cg, cnn::expr::rectify(merge2.get_output(&cg, seg_ij, dur)));
}


ZerothOrderSimpleConcateSemiCRFBuilder::ZerothOrderSimpleConcateSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  : ZerothOrderSemiCRFBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, pretrained),
  seg_emb(dim_hidden1),
  merge2(&m, dim_seg, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1) {
}

void ZerothOrderSimpleConcateSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderSimpleConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, bool train) {
  auto seg_ij = seg_emb(i, j - 1);
  auto dur = dur_emb.embed(j - i);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(cnn::expr::rectify(merge2.get_output(&cg, seg_ij, dur)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, cnn::expr::rectify(merge2.get_output(&cg, seg_ij, dur)));
  }
}


FirstOrderSemiCRFBuilder::FirstOrderSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  float dr,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embeddings)
  :
  input_layer(&m, size_char, dim_char, 0, 0, size_pretrained, dim_pretrained, lstm_input_dim, pretrained_embeddings),
  bilstm_layer(&m, n_layers, lstm_input_dim, dim_hidden1),
  merge2(&m, dim_hidden1, dim_hidden1, dim_hidden1),
  dur_emb(m, dim_duration),
  dropout_rate(dr),
  pretrained(pretrained_embeddings) {

}


cnn::expr::Expression FirstOrderSemiCRFBuilder::supervised_loss(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Segmentation& correct,
  unsigned max_seg_len) {
  unsigned len = sentence.size();
  std::vector<std::vector<bool>> is_ref(len, std::vector<bool>(len + 1, false));

  unsigned cur = 0;
  for (unsigned ri = 0; ri < correct.size(); ++ri) {
    BOOST_ASSERT_MSG(cur < len, "segment index greater than sentence length.");
    unsigned dur = correct[ri];
    if (max_seg_len && dur > max_seg_len) {
      _ERROR << "max_seg_len=" << max_seg_len << " but reference duration is " << dur;
      abort();
    }
    unsigned j = cur + dur;
    BOOST_ASSERT_MSG(j <= len, "End of segment is greater than the input sentence.");

    is_ref[cur][j] = true;
    cur = j;
  }
  BOOST_ASSERT_MSG(cur == len, "Senetence is not consumed by the input span.");

  dur_emb.new_graph(cg);
  bilstm_layer.new_graph(&cg);
  bilstm_layer.set_dropout(dropout_rate);

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned rwid = raw_sentence[i];
    inputs[i] = input_layer.add_input(&cg, wid, 0, rwid);
  }
    
  bilstm_layer.add_inputs(&cg, inputs);
  std::vector<BidirectionalLSTMLayer::Output> hiddens1;
  bilstm_layer.get_outputs(&cg, hiddens1);

  std::vector<cnn::expr::Expression> c(len);
  for (unsigned i = 0; i < len; ++i) {
    c[i] = cnn::expr::rectify(
      merge2.get_output(&cg, hiddens1[i].first, hiddens1[i].second));
  }
  construct_chart(cg, c, max_seg_len, true);

  std::vector<std::vector<cnn::expr::Expression>> alpha(len, std::vector<cnn::expr::Expression>(len + 1));
  std::vector<cnn::expr::Expression> f;
  cnn::expr::Expression ref_alpha;
  for (unsigned k = 1; k <= len; ++k) {
    unsigned j_start = max_seg_len ? (k < max_seg_len ? 0 : k - max_seg_len) : 0;
    for (unsigned j = j_start; j < k; ++j) {
      f.clear();
      if (j == 0) {
        cnn::expr::Expression p = factor_score(cg, k, j, true);
        f.push_back(p);
        if (is_ref[j][k]) { ref_alpha = p; }
      } else {
        unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
        for (unsigned i = i_start; i < j; ++i) {
          cnn::expr::Expression p = factor_score(cg, k, j, i, true);
          f.push_back(p + alpha[i][j]);
          if (is_ref[i][j] && is_ref[j][k]) { ref_alpha = ref_alpha + p; }
        }
      }
      alpha[j][k] = cnn::expr::logsumexp(f);
    }
  }
  f.clear();
  unsigned j_start = max_seg_len ? (len < max_seg_len ? 0 : len - max_seg_len) : 0;
  for (unsigned j = j_start; j < len; ++j) {
    f.push_back(alpha[j][len]);
  }
  return cnn::expr::logsumexp(f) - ref_alpha;
}


void FirstOrderSemiCRFBuilder::viterbi(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  Corpus::Segmentation& yz_pred,
  unsigned max_seg_len) {
  yz_pred.clear();
  unsigned len = sentence.size();

  dur_emb.new_graph(cg);

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned rwid = raw_sentence[i];
    inputs[i] = input_layer.add_input(&cg, wid, 0, rwid);
  }

  bilstm_layer.new_graph(&cg);
  bilstm_layer.add_inputs(&cg, inputs);
  std::vector<BidirectionalLSTMLayer::Output> hiddens1;
  bilstm_layer.get_outputs(&cg, hiddens1);

  std::vector<cnn::expr::Expression> c(len);
  for (unsigned i = 0; i < len; ++i) {
    c[i] = cnn::expr::rectify(
      merge2.get_output(&cg, hiddens1[i].first, hiddens1[i].second));
  }
  construct_chart(cg, c, max_seg_len, false);

  std::vector<std::vector<double>> alpha(len, std::vector<double>(len + 1, 0.));
  std::vector<std::vector<unsigned>> path(len, std::vector<unsigned>(len + 1, 0));
  std::vector<double> f;
  std::vector<unsigned> ijt;
  
  for (unsigned k = 1; k <= len; ++k) {
    unsigned j_start = max_seg_len ? (k < max_seg_len ? 0 : k - max_seg_len) : 0;
    for (unsigned j = j_start; j < k; ++j) {
      f.clear();
      ijt.clear();

      if (j == 0) {
        cnn::expr::Expression p = factor_score(cg, k, j, false);
        double score = cnn::as_scalar(cg.get_value(p));
        f.push_back(score);
        ijt.push_back(0);
      } else {
        unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
        for (unsigned i = i_start; i < j; ++i) {
          cnn::expr::Expression p = factor_score(cg, k, j, i, false);
          double score = cnn::as_scalar(cg.get_value(p));
          f.push_back(score + alpha[i][j]);
          ijt.push_back(i);
        }
      }
      double max_val = f[0]; unsigned max_id = 0;
      for (unsigned i = 1; i < f.size(); ++i) {
        if (max_val < f[i]) { max_val = f[i];  max_id = i; }
      }
      alpha[j][k] = max_val;
      path[j][k] = ijt[max_id];
    }
  }
  unsigned j_start = max_seg_len ? (len < max_seg_len ? 0 : len - max_seg_len) : 0;
  double max_val = alpha[j_start][len]; unsigned max_j = j_start;
  for (unsigned j = j_start + 1; j < len; ++j) {
    if (max_val < alpha[j][len]) { max_val = alpha[j][len]; max_j = j; }
  }
  //yz_pred.push_back(len - max_j);
  auto cur_j = len;
  auto cur_i = max_j;
  while (cur_j > 0) {
    yz_pred.push_back(cur_j - cur_i);
    auto tmp_i = cur_i;
    auto tmp_j = cur_j;
    cur_j = cur_i;
    cur_i = path[tmp_i][tmp_j];
  }
  std::reverse(yz_pred.begin(), yz_pred.end());
}

FirstOrderRnnSemiCRFBuilder::FirstOrderRnnSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_hidden2,
  unsigned dim_duration,
  unsigned dim_seg,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  : 
  FirstOrderSemiCRFBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, pretrained),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge6(&m, dim_seg, dim_seg, dim_duration, dim_seg, dim_seg, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  p_fw_seg_guard(m.add_parameters({ dim_seg })),
  p_bw_seg_guard(m.add_parameters({ dim_seg })),
  p_dur_guard(m.add_parameters({ dim_duration })) {

}

void FirstOrderRnnSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression FirstOrderRnnSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned k, unsigned j, bool train) {
  assert(j == 0);
  const auto& seg_jk = seg_emb(j, k - 1);
  const auto& dur_jk = dur_emb.embed(k - j);
  if (train) {
    return dense.get_output(&cg, 
      cnn::expr::dropout(
      cnn::expr::rectify(merge6.get_output(&cg, 
      cnn::expr::parameter(cg, p_fw_seg_guard), cnn::expr::parameter(cg, p_bw_seg_guard), 
      cnn::expr::parameter(cg, p_dur_guard),
      seg_jk.first, seg_jk.second, dur_jk)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, cnn::expr::rectify(
      merge6.get_output(&cg,
      cnn::expr::parameter(cg, p_fw_seg_guard),
      cnn::expr::parameter(cg, p_bw_seg_guard),
      cnn::expr::parameter(cg, p_dur_guard),
      seg_jk.first, seg_jk.second, dur_jk)));
  }
}

cnn::expr::Expression FirstOrderRnnSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned k, unsigned j, unsigned i, bool train) {
  auto seg_jk = seg_emb(j, k - 1);
  auto dur_jk = dur_emb.embed(k - j);
  auto seg_ij = seg_emb(i, j - 1);
  auto dur_ij = dur_emb.embed(j - i);
  if (train) {
    return dense.get_output(&cg, 
      cnn::expr::dropout(
      cnn::expr::rectify(
      merge6.get_output(&cg,
      seg_ij.first, seg_ij.second, dur_ij,
      seg_jk.first, seg_jk.second, dur_jk)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, cnn::expr::rectify(
      merge6.get_output(&cg,
      seg_ij.first, seg_ij.second, dur_ij,
      seg_jk.first, seg_jk.second, dur_jk)));
  }
}


FirstOrderConcateSemiCRFBuilder::FirstOrderConcateSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg, // should be equal to dim_hidden1 * max_seg_len
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  :
  FirstOrderSemiCRFBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, pretrained),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge4(&m, dim_seg, dim_duration, dim_seg, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  p_seg_guard(m.add_parameters({ dim_seg })),
  p_dur_guard(m.add_parameters({ dim_duration })) {

}

void FirstOrderConcateSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression FirstOrderConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned k, unsigned j, unsigned i, bool train) {
  auto seg_jk = seg_emb(j, k - 1);
  auto dur_jk = dur_emb.embed(k - j);
  auto seg_ij = seg_emb(i, j - 1);
  auto dur_ij = dur_emb.embed(j - i);
  if (train) {
    return dense.get_output(&cg, cnn::expr::dropout(
      cnn::expr::rectify(
      merge4.get_output(&cg, seg_ij, dur_ij, seg_jk, dur_jk)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, cnn::expr::rectify(
      merge4.get_output(&cg, seg_ij, dur_ij, seg_jk, dur_jk)));
  }
}

cnn::expr::Expression FirstOrderConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned k, unsigned j, bool train) {
  assert(j == 0);
  auto seg_jk = seg_emb(j, k - 1);
  auto dur_jk = dur_emb.embed(k - j);
  if (train) {
    return dense.get_output(&cg, 
      cnn::expr::dropout(
      cnn::expr::rectify(
      merge4.get_output(&cg,
      cnn::expr::parameter(cg, p_seg_guard), cnn::expr::parameter(cg, p_dur_guard),
      seg_jk, dur_jk)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, cnn::expr::rectify(
      merge4.get_output(&cg,
      cnn::expr::parameter(cg, p_seg_guard),
      cnn::expr::parameter(cg, p_dur_guard),
      seg_jk, dur_jk)));
  }  
}

FirstOrderSimpleConcateSemiCRFBuilder::FirstOrderSimpleConcateSemiCRFBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg, // should be equal to dim_hidden1 * max_seg_len
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  :
  FirstOrderSemiCRFBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, pretrained),
  seg_emb(dim_hidden1),
  merge4(&m, dim_seg, dim_duration, dim_seg, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  p_seg_guard(m.add_parameters({ dim_seg })),
  p_dur_guard(m.add_parameters({ dim_duration })) {

}

void FirstOrderSimpleConcateSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression FirstOrderSimpleConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned k, unsigned j, unsigned i, bool train) {
  auto seg_jk = seg_emb(j, k - 1);
  auto dur_jk = dur_emb.embed(k - j);
  auto seg_ij = seg_emb(i, j - 1);
  auto dur_ij = dur_emb.embed(j - i);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij, dur_ij, seg_jk, dur_jk)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, 
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij, dur_ij, seg_jk, dur_jk)));
  }
}

cnn::expr::Expression FirstOrderSimpleConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned k, unsigned j, bool train) {
  assert(j == 0);
  auto seg_jk = seg_emb(j, k - 1);
  auto dur_jk = dur_emb.embed(k - j);
  if (train) {
    return dense.get_output(&cg, cnn::expr::dropout(
      cnn::expr::rectify(merge4.get_output(&cg,
      cnn::expr::parameter(cg, p_seg_guard), cnn::expr::parameter(cg, p_dur_guard),
      seg_jk, dur_jk)),
      dropout_rate));
  } else {
    return dense.get_output(&cg, cnn::expr::rectify(
      merge4.get_output(&cg, 
      cnn::expr::parameter(cg, p_seg_guard), cnn::expr::parameter(cg, p_dur_guard),
      seg_jk, dur_jk)));
  }
}

ZerothOrderSemiCRFwLexBuilder::ZerothOrderSemiCRFwLexBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  float dr,
  const std::unordered_map<HashVector, unsigned>& lexicon_,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embeddings) 
  :
  input_layer(&m, size_char, dim_char, 0, 0, size_pretrained, dim_pretrained, lstm_input_dim, pretrained_embeddings),
  bilstm_layer(&m, n_layers, lstm_input_dim, dim_hidden1),
  merge2(&m, dim_hidden1, dim_hidden1, dim_hidden1),
  dur_emb(m, dim_duration),
  dropout_rate(dr),
  lexicon(lexicon_),
  pretrained(pretrained_embeddings) {

}

cnn::expr::Expression ZerothOrderSemiCRFwLexBuilder::supervised_loss(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Segmentation& correct,
  unsigned max_seg_len) {
  unsigned len = sentence.size();
  //
  std::vector<std::vector<bool>> is_ref(len, std::vector<bool>(len + 1, false));

  unsigned cur = 0;
  for (unsigned ri = 0; ri < correct.size(); ++ri) {
    // The beginning position is derivated from training data.
    BOOST_ASSERT_MSG(cur < len, "segment index greater than sentence length.");
    unsigned dur = correct[ri];
    if (max_seg_len && dur > max_seg_len) {
      _ERROR << "max_seg_len=" << max_seg_len << " but reference duration is " << dur;
      abort();
    }
    unsigned j = cur + dur;
    BOOST_ASSERT_MSG(j <= len, "End of segment is greater than the input sentence.");
    is_ref[cur][j] = true;
    cur = j;
  }
  BOOST_ASSERT_MSG(cur == len, "Senetence is not consumed by the input span.");

  dur_emb.new_graph(cg);
  bilstm_layer.new_graph(&cg);
  bilstm_layer.set_dropout(dropout_rate);
  new_graph(cg);
  
  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned rwid = raw_sentence[i];
    inputs[i] = input_layer.add_input(&cg, wid, 0, rwid);
  }

  bilstm_layer.add_inputs(&cg, inputs);
  std::vector<BidirectionalLSTMLayer::Output> hiddens1;
  bilstm_layer.get_outputs(&cg, hiddens1);

  std::vector<cnn::expr::Expression> c(len);
  for (unsigned i = 0; i < len; ++i) {
    c[i] = cnn::expr::rectify(
      merge2.get_output(&cg, hiddens1[i].first, hiddens1[i].second));
  }
  construct_chart(cg, c, max_seg_len, true);

  // f is the expression of overall matrix, fr is the expression of reference.
  std::vector <cnn::expr::Expression> alpha(len + 1), ref_alpha(len + 1);
  std::vector<cnn::expr::Expression> f, fr;
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    fr.clear();
    unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
    for (unsigned i = i_start; i < j; ++i) {
      bool matches_ref = is_ref[i][j];
      unsigned l = get_lexicon_feature(raw_sentence, i, j);
      cnn::expr::Expression p = factor_score(cg, i, j, l, true);

      if (i == 0) {
        f.push_back(p);
        if (matches_ref) { ref_alpha[j] = p; }
      } else {
        f.push_back(p + alpha[i]);
        if (matches_ref) { ref_alpha[j] = p + ref_alpha[i]; }
      }
    }
    alpha[j] = cnn::expr::logsumexp(f);
    // if (fr.size()) ref_alpha[j] = cnn::expr::logsumexp(fr);
  }
  return alpha.back() - ref_alpha.back();
}


void ZerothOrderSemiCRFwLexBuilder::viterbi(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  Corpus::Segmentation& yz_pred,
  unsigned max_seg_len) {

  yz_pred.clear();
  unsigned len = sentence.size();

  dur_emb.new_graph(cg);
  bilstm_layer.new_graph(&cg);
  bilstm_layer.disable_dropout();
  new_graph(cg);

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned rwid = raw_sentence[i];
    inputs[i] = input_layer.add_input(&cg, wid, 0, rwid);
  }

  bilstm_layer.add_inputs(&cg, inputs);
  std::vector<BidirectionalLSTMLayer::Output> hiddens1;
  bilstm_layer.get_outputs(&cg, hiddens1);

  std::vector<cnn::expr::Expression> c(len);
  for (unsigned i = 0; i < len; ++i) {
    c[i] = cnn::expr::rectify(
      merge2.get_output(&cg, hiddens1[i].first, hiddens1[i].second));
  }
  construct_chart(cg, c, max_seg_len, false);

  std::vector<double> alpha(len + 1);
  std::vector<double> f;

  std::vector<std::pair<unsigned, unsigned>> ijt;
  std::vector<std::pair<unsigned, unsigned>> it;

  it.push_back(std::make_pair(0, 0));
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    ijt.clear();
    unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
    for (unsigned i = i_start; i < j; ++i) {
      unsigned l = get_lexicon_feature(raw_sentence, i, j);
      cnn::expr::Expression p = factor_score(cg, i, j, l, false);

      double p_value = cnn::as_scalar(cg.get_value(p));
      if (i == 0) {
        f.push_back(p_value);
      } else {
        f.push_back(p_value + alpha[i]);
      }
      ijt.push_back(std::make_pair(i, j));
    }
    unsigned max_id = 0;
    auto max_val = f[0];
    for (unsigned id = 1; id < f.size(); ++id) {
      auto val = f[id];
      if (max_val < val) { max_val = val; max_id = id; }
    }
    alpha[j] = f[max_id];
    it.push_back(ijt[max_id]);
  }

  auto cur_j = len;
  while (cur_j > 0) {
    auto cur_i = std::get<0>(it[cur_j]);
    yz_pred.push_back(cur_j - cur_i);
    cur_j = cur_i;
  }
  std::reverse(yz_pred.begin(), yz_pred.end());
}

unsigned ZerothOrderSemiCRFwLexBuilder::get_lexicon_feature(const Corpus::Sentence& raw_sentence,
  unsigned i, unsigned j) {
  HashVector key;
  for (unsigned k = i; k < j; ++k) { key.push_back(raw_sentence[k]); }
  unsigned lex = 0;
  auto found = lexicon.find(key);
  if (found != lexicon.end()) {
    lex = found->second;
  }
  return lex;
}


ZerothOrderRnnSemiCRFwRichLexBuilder::ZerothOrderRnnSemiCRFwRichLexBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_lex, unsigned dim_lex,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_hidden2,
  unsigned dim_duration,
  unsigned dim_seg,
  float dropout_rate,
  const std::unordered_map<HashVector, unsigned>& lexicon,
  std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  :
  ZerothOrderSemiCRFwLexBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, lexicon, pretrained),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge4(&m, dim_seg, dim_seg, dim_lex, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  lex_emb(m, size_lex, dim_lex) {
  for (auto& payload : lexicon) {
    assert(payload.second < entities_embedding.size());
    lex_emb.p_labels->Initialize(payload.second, entities_embedding[payload.second]);
  }
}

void ZerothOrderRnnSemiCRFwRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderRnnSemiCRFwRichLexBuilder::factor_score(
  cnn::ComputationGraph& cg, unsigned i, unsigned j, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& lex = lex_emb.embed(l);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur)));
  }
}

void ZerothOrderRnnSemiCRFwRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}


ZerothOrderConcateSemiCRFwRichLexBuilder::ZerothOrderConcateSemiCRFwRichLexBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_lex, unsigned dim_lex,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<HashVector, unsigned>& lexicon,
  std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained)
  :
  ZerothOrderSemiCRFwLexBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, lexicon, pretrained),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge3(&m, dim_seg, dim_lex, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  lex_emb(m, size_lex, dim_lex) {
  for (auto& payload : lexicon) {
    assert(payload.second < entities_embedding.size());
    lex_emb.p_labels->Initialize(payload.second, entities_embedding[payload.second]);
  }
}

void ZerothOrderConcateSemiCRFwRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderConcateSemiCRFwRichLexBuilder::factor_score(
  cnn::ComputationGraph& cg, unsigned i, unsigned j, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& lex = lex_emb.embed(l);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge3.get_output(&cg, seg_ij, lex, dur)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge3.get_output(&cg, seg_ij, lex, dur)));
  }
}

void ZerothOrderConcateSemiCRFwRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

ZerothOrderRnnSemiCRFwTuneRichLexBuilder::ZerothOrderRnnSemiCRFwTuneRichLexBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_lex, unsigned dim_lex,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_hidden2,
  unsigned dim_duration,
  unsigned dim_seg,
  float dropout_rate,
  const std::unordered_map<HashVector, unsigned>& lexicon,
  std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained,
  bool init_segment_embedding)
  :
  ZerothOrderSemiCRFwLexBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, lexicon, pretrained),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge4(&m, dim_seg, dim_seg, dim_lex, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  lex_emb(m, size_lex, dim_lex) {
  if (init_segment_embedding) {
    for (auto& payload : lexicon) {
      assert(payload.second < entities_embedding.size());
      if (payload.second != 0) {
        lex_emb.p_labels->Initialize(payload.second, entities_embedding[payload.second]);
      }
    }
  }
}

void ZerothOrderRnnSemiCRFwTuneRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderRnnSemiCRFwTuneRichLexBuilder::factor_score(
  cnn::ComputationGraph& cg, unsigned i, unsigned j, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& lex = lex_emb.embed(l);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur)));
  }
}

void ZerothOrderRnnSemiCRFwTuneRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}


ZerothOrderConcateSemiCRFwTuneRichLexBuilder::ZerothOrderConcateSemiCRFwTuneRichLexBuilder(cnn::Model& m,
  unsigned size_char, unsigned dim_char,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_lex, unsigned dim_lex,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<HashVector, unsigned>& lexicon,
  std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained,
  bool init_segment_embedding)
  :
  ZerothOrderSemiCRFwLexBuilder(m, size_char, dim_char, size_pretrained, dim_pretrained,
  n_layers, lstm_input_dim, dim_hidden1, dim_duration, dropout_rate, lexicon, pretrained),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge3(&m, dim_seg, dim_lex, dim_duration, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  lex_emb(m, size_lex, dim_lex) {
  if (init_segment_embedding) {
    for (auto& payload : lexicon) {
      assert(payload.second < entities_embedding.size());
      if (payload.second != 0) {
        lex_emb.p_labels->Initialize(payload.second, entities_embedding[payload.second]);
      }
    }
  }
}

void ZerothOrderConcateSemiCRFwTuneRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ZerothOrderConcateSemiCRFwTuneRichLexBuilder::factor_score(
  cnn::ComputationGraph& cg, unsigned i, unsigned j, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& lex = lex_emb.embed(l);
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge3.get_output(&cg, seg_ij, lex, dur)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge3.get_output(&cg, seg_ij, lex, dur)));
  }
}

void ZerothOrderConcateSemiCRFwTuneRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

