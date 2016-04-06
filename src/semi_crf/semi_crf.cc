#include "semi_crf.h"
#include "logging.h"
#include "utils.h"
#include <cstdio>
#include <fstream>
#include <boost/algorithm/string.hpp>


SemiCRFBuilder::SemiCRFBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  float dr,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  input_layer(&m, size_word, dim_word, size_pos, dim_pos, size_pretrained, dim_pretrained, lstm_input_dim, pretrained_embed),
  bilstm_layer(&m, n_layers, lstm_input_dim, dim_hidden1),
  merge2(&m, dim_hidden1, dim_hidden1, dim_hidden1),
  y_emb(m, size_tag_, dim_tag),
  dur_emb(m, dim_duration),
  pretrained(pretrained_embed),
  size_tag(size_tag_),
  dropout_rate(dr) {

}


cnn::expr::Expression SemiCRFBuilder::supervised_loss(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Sentence& postag,
  const Corpus::Segmentation& correct,
  unsigned max_seg_len) {
  unsigned len = sentence.size();
  //
  std::vector<std::vector<std::vector<bool>>> is_ref(len,
    std::vector<std::vector<bool>>(len + 1, std::vector<bool>(size_tag, false)));

  unsigned cur = 0;
  for (unsigned ri = 0; ri < correct.size(); ++ri) {
    // The beginning position is derivated from training data.
    BOOST_ASSERT_MSG(cur < len, "segment index greater than sentence length.");
    unsigned y = correct[ri].first;
    unsigned dur = correct[ri].second;
    if (max_seg_len && dur > max_seg_len) {
      _ERROR << "max_seg_len=" << max_seg_len << " but reference duration is " << dur;
      abort();
    }
    unsigned j = cur + dur;
    BOOST_ASSERT_MSG(j <= len, "End of segment is greater than the input sentence.");
    is_ref[cur][j][y] = true;
    cur = j;
  }
  BOOST_ASSERT_MSG(cur == len, "Senetence is not consumed by the input span.");

  y_emb.new_graph(cg);
  dur_emb.new_graph(cg);
  bilstm_layer.set_dropout(dropout_rate);

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned pid = postag[i];
    unsigned rwid = raw_sentence[i];
    if (!pretrained.count(rwid)) { rwid = 0; }
    inputs[i] = input_layer.add_input(&cg, wid, pid, rwid);
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

  construct_chart(cg, c, max_seg_len, true);
  std::vector <cnn::expr::Expression> alpha(len + 1), ref_alpha(len + 1);
  std::vector<cnn::expr::Expression> f;
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    for (unsigned tag = 0; tag < size_tag; ++tag) {
      unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
      for (unsigned i = i_start; i < j; ++i) {
        bool matches_ref = is_ref[i][j][tag];
        cnn::expr::Expression p = factor_score(cg, i, j, tag, true);
        
        if (i == 0) {
          f.push_back(p);
          if (matches_ref) { ref_alpha[j] = p; }
        } else {
          f.push_back(p + alpha[i]);
          if (matches_ref) { ref_alpha[j] = p + ref_alpha[i]; }
        }
      }
    }
    alpha[j] = cnn::expr::logsumexp(f);
  }
  return alpha.back() - ref_alpha.back();
}

cnn::expr::Expression SemiCRFBuilder::margin_loss(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Sentence& postag,
  const Corpus::Segmentation& correct,
  unsigned max_seg_len) {
  unsigned len = sentence.size();
  //
  std::vector<std::vector<std::vector<bool>>> is_ref(len,
    std::vector<std::vector<bool>>(len + 1, std::vector<bool>(size_tag, false)));

  unsigned cur = 0;
  for (unsigned ri = 0; ri < correct.size(); ++ri) {
    // The beginning position is derivated from training data.
    BOOST_ASSERT_MSG(cur < len, "segment index greater than sentence length.");
    unsigned y = correct[ri].first;
    unsigned dur = correct[ri].second;
    if (max_seg_len && dur > max_seg_len) {
      _ERROR << "max_seg_len=" << max_seg_len << " but reference duration is " << dur;
      abort();
    }
    unsigned j = cur + dur;
    BOOST_ASSERT_MSG(j <= len, "End of segment is greater than the input sentence.");
    is_ref[cur][j][y] = true;
    cur = j;
  }
  BOOST_ASSERT_MSG(cur == len, "Senetence is not consumed by the input span.");

  y_emb.new_graph(cg);
  dur_emb.new_graph(cg);
  bilstm_layer.set_dropout(dropout_rate);

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned pid = postag[i];
    unsigned rwid = raw_sentence[i];
    if (!pretrained.count(rwid)) { rwid = 0; }
    inputs[i] = input_layer.add_input(&cg, wid, pid, rwid);
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
  construct_chart(cg, c, max_seg_len, true);

  // f is the expression of overall matrix, fr is the expression of reference.
  std::vector<cnn::expr::Expression> alpha(len + 1), ref_alpha(len + 1);
  std::vector<cnn::expr::Expression> f;
  std::vector<std::tuple<unsigned, unsigned, unsigned>> ijt;
  std::vector<std::tuple<unsigned, unsigned, unsigned>> it;

  it.push_back(std::make_tuple(0, 0, 0));
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    for (unsigned tag = 0; tag < size_tag; ++tag) {
      unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
      for (unsigned i = i_start; i < j; ++i) {
        bool matches_ref = is_ref[i][j][tag];
        cnn::expr::Expression p = factor_score(cg, i, j, tag, true);

        if (i == 0) {
          f.push_back(p);
          if (matches_ref) { ref_alpha[j] = p; }
        } else {
          f.push_back(p + alpha[i]);
          if (matches_ref) { ref_alpha[j] = p + ref_alpha[i]; }
        }
        ijt.push_back(std::make_tuple(i, j, tag));
      }
    }
    unsigned max_id = 0;
    double max_val = cnn::as_scalar(cg.get_value(f[0]));
    for (unsigned id = 1; id < f.size(); ++id) {
      auto val = cnn::as_scalar(cg.get_value(f[id]));
      if (max_val < val) { max_val = val; max_id = id; }
    }
    alpha[j] = f[max_id];
    it.push_back(ijt[max_id]);
  }

  Corpus::Segmentation yz_pred;
  auto cur_j = len;
  while (cur_j > 0) {
    auto cur_i = std::get<0>(it[cur_j]);
    yz_pred.push_back(std::make_pair(std::get<2>(it[cur_j]), cur_j - cur_i));
    cur_j = cur_i;
  }
  std::reverse(yz_pred.begin(), yz_pred.end());
  double l = segmentation_loss(correct, yz_pred);
  return cnn::expr::pairwise_rank_loss(cnn::expr::reshape(ref_alpha.back(), { 1 }),
    cnn::expr::reshape(alpha.back(), { 1 }), l);
}


void SemiCRFBuilder::viterbi(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Sentence& postag,
  Corpus::Segmentation& yz_pred,
  unsigned max_seg_len) {

  yz_pred.clear();
  unsigned len = sentence.size();

  y_emb.new_graph(cg);
  dur_emb.new_graph(cg);
  bilstm_layer.disable_dropout();

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned pid = postag[i];
    unsigned rwid = raw_sentence[i];
    if (!pretrained.count(rwid)) { rwid = 0; }
    inputs[i] = input_layer.add_input(&cg, wid, pid, rwid);
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

  std::vector<double> alpha(len + 1);
  std::vector<double> f;

  std::vector<std::tuple<unsigned, unsigned, unsigned>> ijt;
  std::vector<std::tuple<unsigned, unsigned, unsigned>> it;

  it.push_back(std::make_tuple(0, 0, 0));
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    ijt.clear();
    for (unsigned tag = 0; tag < size_tag; ++tag) {
      unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
      for (unsigned i = i_start; i < j; ++i) {
        cnn::expr::Expression p = factor_score(cg, i, j, tag, false);

        double p_value = cnn::as_scalar(cg.get_value(p));
        if (i == 0) {
          f.push_back(p_value);
        } else {
          f.push_back(p_value + alpha[i]);
        }
        ijt.push_back(std::make_tuple(i, j, tag));
      }
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
    yz_pred.push_back(std::make_pair(std::get<2>(it[cur_j]), cur_j - cur_i));
    cur_j = cur_i;
  }
  std::reverse(yz_pred.begin(), yz_pred.end());
}


RNNSemiCRFBuilder::RNNSemiCRFBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  unsigned dim_seg,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
                 dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
                 dim_duration, dropout_rate, pretrained_embed),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge4(&m, dim_seg, dim_seg, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1) {

}

void RNNSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  // if (train) { seg_emb.set_dropout(dropout_rate); } else { seg_emb.disable_dropout(); }
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression RNNSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  if (train) {
    return dense.get_output(&cg,
        cnn::expr::dropout(
          cnn::expr::rectify(merge4.get_output(&cg, seg_ij.first, seg_ij.second, dur, y)),
          dropout_rate));
  } else {
     return dense.get_output(&cg,
          cnn::expr::rectify(merge4.get_output(&cg, seg_ij.first, seg_ij.second, dur, y)));
  }
}

CNNSemiCRFBuilder::CNNSemiCRFBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  const std::vector<std::pair<unsigned, unsigned>>& filters,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, pretrained_embed),
  seg_emb(m, dim_hidden1, filters),
  merge3(&m, dim_seg, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1) {

}

void CNNSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression CNNSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  return dense.get_output(&cg, cnn::expr::rectify(merge3.get_output(&cg, seg_ij, dur, y)));
}


ConcateSemiCRFBuilder::ConcateSemiCRFBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, pretrained_embed),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge3(&m, dim_seg, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1) {

}

void ConcateSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  return dense.get_output(&cg, cnn::expr::rectify(merge3.get_output(&cg, seg_ij, dur, y)));
}

SimpleConcateSemiCRFBuilder::SimpleConcateSemiCRFBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned max_seg_len,
  unsigned dim_seg,
  unsigned dim_duration,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, pretrained_embed),
  seg_emb(dim_hidden1),
  merge3(&m, dim_seg, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1) {

}

void SimpleConcateSemiCRFBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression SimpleConcateSemiCRFBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  return dense.get_output(&cg, cnn::expr::rectify(merge3.get_output(&cg, seg_ij, dur, y)));
}

SemiCRFwLexBuilder::SemiCRFwLexBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  float dr,
  const std::unordered_map<HashVector, unsigned>& lexicon_,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_) 
  :
  input_layer(&m, size_word, dim_word, size_pos, dim_pos, size_pretrained, dim_pretrained, lstm_input_dim, pretrained_),
  bilstm_layer(&m, n_layers, lstm_input_dim, dim_hidden1),
  merge2(&m, dim_hidden1, dim_hidden1, dim_hidden1),
  y_emb(m, size_tag_, dim_tag),
  // lex_emb(m, size_lex, dim_lex),
  dur_emb(m, dim_duration),
  pretrained(pretrained_),
  lexicon(lexicon_),
  size_tag(size_tag_),
  dropout_rate(dr) {

}

cnn::expr::Expression SemiCRFwLexBuilder::supervised_loss(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Sentence& postag,
  const Corpus::Segmentation& correct,
  unsigned max_seg_len) {
  unsigned len = sentence.size();
  //
  std::vector<std::vector<std::vector<bool>>> is_ref(len,
    std::vector<std::vector<bool>>(len + 1, std::vector<bool>(size_tag, false)));

  unsigned cur = 0;
  for (unsigned ri = 0; ri < correct.size(); ++ri) {
    // The beginning position is derivated from training data.
    BOOST_ASSERT_MSG(cur < len, "segment index greater than sentence length.");
    unsigned y = correct[ri].first;
    unsigned dur = correct[ri].second;
    if (max_seg_len && dur > max_seg_len) {
      _ERROR << "max_seg_len=" << max_seg_len << " but reference duration is " << dur;
      abort();
    }
    unsigned j = cur + dur;
    BOOST_ASSERT_MSG(j <= len, "End of segment is greater than the input sentence.");
    is_ref[cur][j][y] = true;
    cur = j;
  }
  BOOST_ASSERT_MSG(cur == len, "Senetence is not consumed by the input span.");

  y_emb.new_graph(cg);
  dur_emb.new_graph(cg);
  bilstm_layer.set_dropout(dropout_rate);
  new_graph(cg);

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned pid = postag[i];
    unsigned rwid = raw_sentence[i];
    if (!pretrained.count(rwid)) { rwid = 0; }
    inputs[i] = input_layer.add_input(&cg, wid, pid, rwid);
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

  construct_chart(cg, c, max_seg_len, true);
  std::vector <cnn::expr::Expression> alpha(len + 1), ref_alpha(len + 1);
  std::vector<cnn::expr::Expression> f;
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    for (unsigned tag = 0; tag < size_tag; ++tag) {
      unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
      for (unsigned i = i_start; i < j; ++i) {
        bool matches_ref = is_ref[i][j][tag];
        unsigned lex = get_lexicon_feature(raw_sentence, i, j);
        cnn::expr::Expression p = factor_score(cg, i, j, tag, lex, true);

        if (i == 0) {
          f.push_back(p);
          if (matches_ref) { ref_alpha[j] = p; }
        } else {
          f.push_back(p + alpha[i]);
          if (matches_ref) { ref_alpha[j] = p + ref_alpha[i]; }
        }
      }
    }
    alpha[j] = cnn::expr::logsumexp(f);
  }
  return alpha.back() - ref_alpha.back();
}

cnn::expr::Expression SemiCRFwLexBuilder::margin_loss(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Sentence& postag,
  const Corpus::Segmentation& correct,
  unsigned max_seg_len) {
  BOOST_ASSERT_MSG(false, "not implemented.");
  return cnn::expr::Expression();
}

void SemiCRFwLexBuilder::viterbi(cnn::ComputationGraph& cg,
  const Corpus::Sentence& raw_sentence,
  const Corpus::Sentence& sentence,
  const Corpus::Sentence& postag,
  Corpus::Segmentation& yz_pred,
  unsigned max_seg_len) {

  yz_pred.clear();
  unsigned len = sentence.size();

  y_emb.new_graph(cg);
  dur_emb.new_graph(cg);
  new_graph(cg);
  bilstm_layer.disable_dropout();

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned pid = postag[i];
    unsigned rwid = raw_sentence[i];
    if (!pretrained.count(rwid)) { rwid = 0; }
    inputs[i] = input_layer.add_input(&cg, wid, pid, rwid);
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

  std::vector<double> alpha(len + 1);
  std::vector<double> f;

  std::vector<std::tuple<unsigned, unsigned, unsigned>> ijt;
  std::vector<std::tuple<unsigned, unsigned, unsigned>> it;

  it.push_back(std::make_tuple(0, 0, 0));
  for (unsigned j = 1; j <= len; ++j) {
    f.clear();
    ijt.clear();
    for (unsigned tag = 0; tag < size_tag; ++tag) {
      unsigned i_start = max_seg_len ? (j < max_seg_len ? 0 : j - max_seg_len) : 0;
      for (unsigned i = i_start; i < j; ++i) {
        unsigned lex = get_lexicon_feature(raw_sentence, i, j);
        cnn::expr::Expression p = factor_score(cg, i, j, tag, lex, false);

        double p_value = cnn::as_scalar(cg.get_value(p));
        if (i == 0) {
          f.push_back(p_value);
        } else {
          f.push_back(p_value + alpha[i]);
        }
        ijt.push_back(std::make_tuple(i, j, tag));
      }
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
    yz_pred.push_back(std::make_pair(std::get<2>(it[cur_j]), cur_j - cur_i));
    cur_j = cur_i;
  }
  std::reverse(yz_pred.begin(), yz_pred.end());
}


unsigned SemiCRFwLexBuilder::get_lexicon_feature(const Corpus::Sentence& raw_sentence,
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

RNNSemiCRFwLexBuilder::RNNSemiCRFwLexBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned size_lex, unsigned dim_lex,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  unsigned dim_seg,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<HashVector, unsigned>& lexicon,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFwLexBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, lexicon, pretrained_embed),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge5(&m, dim_seg, dim_seg, dim_lex, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1), lex_emb(m, size_lex, dim_lex) {

}

void RNNSemiCRFwLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

void RNNSemiCRFwLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  // if (train) { seg_emb.set_dropout(dropout_rate); } else { seg_emb.disable_dropout(); }
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression RNNSemiCRFwLexBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  const auto& lex = lex_emb.embed(l);
  if (train) {
    return dense.get_output(&cg,
        cnn::expr::dropout(
          cnn::expr::rectify(merge5.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur, y)),
          dropout_rate));
  } else {
    return dense.get_output(&cg,
        cnn::expr::rectify(merge5.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur, y)));
  }
}

ConcateSemiCRFwLexBuilder::ConcateSemiCRFwLexBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
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
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFwLexBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, lexicon, pretrained_embed),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge4(&m, dim_seg, dim_lex, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1), lex_emb(m, size_lex, dim_lex) {

}

void ConcateSemiCRFwLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

void ConcateSemiCRFwLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

cnn::expr::Expression ConcateSemiCRFwLexBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  const auto& lex = lex_emb.embed(l);
  if (train) {
    return dense.get_output(&cg,
        cnn::expr::dropout(
          cnn::expr::rectify(merge4.get_output(&cg, seg_ij, lex, dur, y)),
          dropout_rate));
  } else {
    return dense.get_output(&cg,
        cnn::expr::rectify(merge4.get_output(&cg, seg_ij, lex, dur, y)));
  }
}


RNNSemiCRFwRichLexBuilder::RNNSemiCRFwRichLexBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned size_lex, unsigned dim_lex,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  unsigned dim_seg,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<HashVector, unsigned>& lexicon,
  const std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFwLexBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, lexicon, pretrained_embed),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge5(&m, dim_seg, dim_seg, dim_lex, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  lex_emb(m, size_lex, dim_lex) {
  for (auto& payload : lexicon) {
    assert(payload.second < entities_embedding.size());
    if (payload.second != 0) {
      lex_emb.p_labels->Initialize(payload.second, entities_embedding[payload.second]);
    }
  }
}

void RNNSemiCRFwRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

void RNNSemiCRFwRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression RNNSemiCRFwRichLexBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  cnn::expr::Expression lex;
  if (l > 0) {
    lex = lex_emb.embed(l);
  } else {
    // For the known entities, fix their embedding, while for the unknown (id=0), tune the parameters.
    lex = cnn::expr::lookup(cg, lex_emb.p_labels, l);
  }
  
  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge5.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur, y)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge5.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur, y)));
  }
}

ConcateSemiCRFwRichLexBuilder::ConcateSemiCRFwRichLexBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
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
  const std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed)
  :
  SemiCRFwLexBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, lexicon, pretrained_embed),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge4(&m, dim_seg, dim_lex, dim_duration, dim_tag, dim_hidden2),
  dense(&m, dim_hidden2, 1),
  lex_emb(m, size_lex, dim_lex) {
  for (auto& payload : lexicon) {
    assert(payload.second < entities_embedding.size());
    if (payload.second != 0) {
      lex_emb.p_labels->Initialize(payload.second, entities_embedding[payload.second]);
    }
  }
}

void ConcateSemiCRFwRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

void ConcateSemiCRFwRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ConcateSemiCRFwRichLexBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  cnn::expr::Expression lex;
  if (l > 0) {
    lex = lex_emb.embed(l);
  } else {
    // For the known entities, fix their embedding, while for the unknown (id=0), tune the parameters.
    lex = cnn::expr::lookup(cg, lex_emb.p_labels, l);
  }

  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij, lex, dur, y)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij, lex, dur, y)));
  }
}


RNNSemiCRFwTuneRichLexBuilder::RNNSemiCRFwTuneRichLexBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
  unsigned size_lex, unsigned dim_lex,
  unsigned n_layers,
  unsigned lstm_input_dim,
  unsigned dim_hidden1,
  unsigned dim_duration,
  unsigned dim_seg,
  unsigned dim_hidden2,
  float dropout_rate,
  const std::unordered_map<HashVector, unsigned>& lexicon,
  const std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed,
  bool init_segment_embedding)
  :
  SemiCRFwLexBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, lexicon, pretrained_embed),
  seg_emb(m, n_layers, dim_hidden1, dim_seg),
  merge5(&m, dim_seg, dim_seg, dim_lex, dim_duration, dim_tag, dim_hidden2),
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

void RNNSemiCRFwTuneRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

void RNNSemiCRFwTuneRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression RNNSemiCRFwTuneRichLexBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  cnn::expr::Expression lex;
  if (l > 0) {
    lex = lex_emb.embed(l);
  } else {
    // For the known entities, fix their embedding, while for the unknown (id=0), tune the parameters.
    lex = cnn::expr::lookup(cg, lex_emb.p_labels, l);
  }

  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge5.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur, y)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge5.get_output(&cg, seg_ij.first, seg_ij.second, lex, dur, y)));
  }
}

ConcateSemiCRFwTuneRichLexBuilder::ConcateSemiCRFwTuneRichLexBuilder(cnn::Model& m,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained, unsigned dim_pretrained,
  unsigned size_tag_, unsigned dim_tag,
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
  const std::vector<std::vector<float>>& entities_embedding,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embed,
  bool init_segment_embedding)
  :
  SemiCRFwLexBuilder(m, size_word, dim_word, size_pos, dim_pos, size_pretrained,
  dim_pretrained, size_tag_, dim_tag, n_layers, lstm_input_dim, dim_hidden1,
  dim_duration, dropout_rate, lexicon, pretrained_embed),
  seg_emb(m, dim_hidden1, dim_seg, max_seg_len),
  merge4(&m, dim_seg, dim_lex, dim_duration, dim_tag, dim_hidden2),
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

void ConcateSemiCRFwTuneRichLexBuilder::new_graph(cnn::ComputationGraph& cg) {
  lex_emb.new_graph(cg);
}

void ConcateSemiCRFwTuneRichLexBuilder::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  unsigned max_seg_len, bool train) {
  seg_emb.construct_chart(cg, c, max_seg_len);
}

cnn::expr::Expression ConcateSemiCRFwTuneRichLexBuilder::factor_score(cnn::ComputationGraph& cg,
  unsigned i, unsigned j, unsigned tag, unsigned l, bool train) {
  const auto& seg_ij = seg_emb(i, j - 1);
  const auto& dur = dur_emb.embed(j - i);
  const auto& y = y_emb.embed(tag);
  cnn::expr::Expression lex;
  if (l > 0) {
    lex = lex_emb.embed(l);
  } else {
    // For the known entities, fix their embedding, while for the unknown (id=0), tune the parameters.
    lex = cnn::expr::lookup(cg, lex_emb.p_labels, l);
  }

  if (train) {
    return dense.get_output(&cg,
      cnn::expr::dropout(
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij, lex, dur, y)),
      dropout_rate));
  } else {
    return dense.get_output(&cg,
      cnn::expr::rectify(merge4.get_output(&cg, seg_ij, lex, dur, y)));
  }
}