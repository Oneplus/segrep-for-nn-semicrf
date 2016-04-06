#include "crf.h"

CRFBuilder::CRFBuilder(cnn::Model* model,
  unsigned size_word, unsigned dim_word,
  unsigned size_pos, unsigned dim_pos,
  unsigned size_pretrained_word, unsigned dim_pretrained_word,
  unsigned size_label, unsigned dim_label,
  unsigned n_layers, unsigned dim_lstm_input, unsigned dim_hidden,
  float dr,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding) : 
  input_layer(model, size_word, dim_word, size_pos, dim_pos, size_pretrained_word, dim_pretrained_word, 
  dim_lstm_input, pretrained_embedding),
  bilstm_layer(model, n_layers, dim_lstm_input, dim_hidden),
  merge_layer(model, dim_hidden, dim_hidden, dim_label, dim_hidden),
  dense_layer(model, dim_hidden, 1), 
  n_labels(size_label),
  dropout_rate(dr),
  p_l(model->add_lookup_parameters(size_label, { dim_label, 1 })),
  p_tran(model->add_lookup_parameters(size_label * size_label, {1})),
  pretrained(pretrained_embedding) {

}

cnn::expr::Expression CRFBuilder::supervised_loss(cnn::ComputationGraph* cg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<unsigned>& postag,
  const std::vector<unsigned>& correct) {
  unsigned len = raw_sentence.size();
  std::vector<ExpressionRow> emit_matrix(len, ExpressionRow(n_labels));
  std::vector<ExpressionRow> tran_matrix(n_labels, ExpressionRow(n_labels));

  std::vector<Expression> uni_labels(n_labels);
  for (unsigned t = 0; t < n_labels; ++t) {
    uni_labels[t] = cnn::expr::lookup(*cg, p_l, t);
    for (unsigned pt = 0; pt < n_labels; ++pt) {
      tran_matrix[pt][t] = cnn::expr::lookup(*cg, p_tran, pt * n_labels + t);
    }
  }

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned pid = postag[i];
    unsigned rwid = raw_sentence[i];
    if (!pretrained.count(rwid)) { rwid = 0; }
    inputs[i] = input_layer.add_input(cg, wid, pid, rwid);
  }
  
  bilstm_layer.new_graph(cg);
  bilstm_layer.set_dropout(dropout_rate);
  bilstm_layer.add_inputs(cg, inputs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bilstm_layer.get_outputs(cg, hidden1);

  for (unsigned i = 0; i < len; ++i) {
    for (unsigned t = 0; t < n_labels; ++t) {
      emit_matrix[i][t] = dense_layer.get_output(cg,
          cnn::expr::dropout(cnn::expr::rectify(
              merge_layer.get_output(cg, hidden1[i].first, hidden1[i].second, uni_labels[t])),
            dropout_rate));
    }
  }

  std::vector<ExpressionRow> alpha(len, ExpressionRow(n_labels));
  std::vector<Expression> path(len);

  for (unsigned i = 0; i < len; ++i) {
    for (unsigned t = 0; t < n_labels; ++t) {
      std::vector<Expression> f;
      if (i == 0) {
        f.push_back(emit_matrix[i][t]);
        if (t == correct[i]) {
          path[i] = emit_matrix[i][t];
        }
      } else {
        for (unsigned pt = 0; pt < n_labels; ++pt) {
          f.push_back(alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t]);
          if (pt == correct[i - 1] && t == correct[i]) {
            path[i] = path[i - 1] + emit_matrix[i][t] + tran_matrix[pt][t];
          }
        }
      }
      alpha[i][t] = cnn::expr::logsumexp(f);
      //if (fr.size() > 0) { path[i] = cnn::expr::logsumexp(fr); }
    }
  }

  std::vector<Expression> f;
  for (unsigned t = 0; t < n_labels; ++t) {
    f.push_back(alpha[len - 1][t]);
  }
  return cnn::expr::logsumexp(f) - path.back();
}

void CRFBuilder::viterbi(cnn::ComputationGraph* cg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<unsigned>& postag,
  std::vector<unsigned>& predict) {
  unsigned len = raw_sentence.size();
  std::vector<std::vector<double>> emit_matrix(len, std::vector<double>(n_labels));
  std::vector<std::vector<double>> tran_matrix(n_labels, std::vector<double>(n_labels));

  std::vector<Expression> uni_labels(n_labels);
  for (unsigned t = 0; t < n_labels; ++t) {
    uni_labels[t] = cnn::expr::lookup(*cg, p_l, t);
    for (unsigned pt = 0; pt < n_labels; ++pt) {
      tran_matrix[pt][t] = cnn::as_scalar(cg->get_value(
        cnn::expr::lookup(*cg, p_tran, pt * n_labels + t)));
    }
  }

  std::vector<Expression> inputs(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned wid = sentence[i];
    unsigned pid = postag[i];
    unsigned rwid = raw_sentence[i];
    if (!pretrained.count(rwid)) { rwid = 0; }
    inputs[i] = input_layer.add_input(cg, wid, pid, rwid);
  }

  bilstm_layer.new_graph(cg);
  bilstm_layer.disable_dropout();
  bilstm_layer.add_inputs(cg, inputs);

  std::vector<BidirectionalLSTMLayer::Output> hidden1;
  bilstm_layer.get_outputs(cg, hidden1);

  for (unsigned i = 0; i < len; ++i) {
    for (unsigned t = 0; t < n_labels; ++t) {
      emit_matrix[i][t] = cnn::as_scalar(cg->get_value(dense_layer.get_output(cg, 
        cnn::expr::rectify(merge_layer.get_output(cg, hidden1[i].first, hidden1[i].second, uni_labels[t])))));
    }
  }

  std::vector<std::vector<double>> alpha(len, std::vector<double>(n_labels));
  std::vector<std::vector<unsigned>> path(len, std::vector<unsigned>(n_labels));

  for (unsigned i = 0; i < len; ++i) {
    for (unsigned t = 0; t < n_labels; ++t) {
      if (i == 0) {
        alpha[i][t] = emit_matrix[i][t];
        path[i][t] = n_labels;
        continue;
      }
      
      for (unsigned pt = 0; pt < n_labels; ++pt) {
        if (pt == 0) {
          alpha[i][t] = alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t];
          path[i][t] = pt;
        } else {
          double score = alpha[i - 1][pt] + emit_matrix[i][t] + tran_matrix[pt][t];
          if (score > alpha[i][t]) {
            alpha[i][t] = score;
            path[i][t] = pt;
          }
        }
      }
    }
  }

  unsigned best = 0; double best_score = alpha[len - 1][0];
  for (unsigned t = 1; t < n_labels; ++t) {
    if (best_score < alpha[len - 1][t]) { best = t; best_score = alpha[len - 1][t]; }
  }
  predict.clear(); predict.push_back(best);
  for (unsigned i = len - 1; i > 0; -- i) {
    best = path[i][best];
    predict.push_back(best);
  }
  std::reverse(predict.begin(), predict.end());
}
