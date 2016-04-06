#include "layer.h"

StaticInputLayer::StaticInputLayer(cnn::Model* model,
  unsigned size_word, unsigned dim_word,
  unsigned size_postag, unsigned dim_postag,
  unsigned size_pretrained_word, unsigned dim_pretrained_word,
  unsigned dim_output,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained) :
  p_w(nullptr), p_p(nullptr), p_t(nullptr),
  p_ib(nullptr), p_w2l(nullptr), p_p2l(nullptr), p_t2l(nullptr),
  use_word(true), use_postag(true), use_pretrained_word(true) {

  p_ib = model->add_parameters({ dim_output, 1 });
  if (dim_word == 0) {
    std::cerr << "Word dim should be greater than 0." << std::endl;
    std::cerr << "Fine-tuned word embedding is inactivated." << std::endl;
    use_word = false;
  } else {
    p_w = model->add_lookup_parameters(size_word, { dim_word, 1 });
    p_w2l = model->add_parameters({ dim_output, dim_word });
  }

  if (dim_postag == 0) {
    std::cerr << "Postag dim should be greater than 0." << std::endl;
    std::cerr << "Fine-tuned postag embedding is inactivated." << std::endl;
    use_postag = false;
  } else {
    p_p = model->add_lookup_parameters(size_postag, { dim_postag, 1 });
    p_p2l = model->add_parameters({ dim_output, dim_postag });
  }

  if (dim_pretrained_word == 0) {
    std::cerr << "Pretrained word embedding dim should be greater than 0." << std::endl;
    std::cerr << "Pretrained word embedding is inactivated." << std::endl;
    use_pretrained_word = false;
  } else {
    p_t = model->add_lookup_parameters(size_pretrained_word, { dim_pretrained_word, 1 });
    for (auto it : pretrained) { p_t->Initialize(it.first, it.second); }
    p_t2l = model->add_parameters({ dim_output, dim_pretrained_word });
  }
}


cnn::expr::Expression StaticInputLayer::add_input(cnn::ComputationGraph* hg,
  unsigned wid, unsigned pid, unsigned pre_wid) {
  cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
  if (use_word && wid > 0) {
    cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
    cnn::expr::Expression w = cnn::expr::lookup(*hg, p_w, wid);
    expr = cnn::expr::affine_transform({ expr, w2l, w });
  }
  if (use_postag && pid > 0) {
    cnn::expr::Expression p2l = cnn::expr::parameter(*hg, p_p2l);
    cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
    expr = cnn::expr::affine_transform({ expr, p2l, p });
  }
  if (use_pretrained_word && pre_wid > 0) {
    cnn::expr::Expression t2l = cnn::expr::parameter(*hg, p_t2l);
    cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
    expr = cnn::expr::affine_transform({ expr, t2l, t });
  }
  return cnn::expr::rectify(expr);
}


DynamicInputLayer::DynamicInputLayer(cnn::Model* model,
  unsigned size_word, unsigned dim_word,
  unsigned size_postag, unsigned dim_postag,
  unsigned size_pretrained_word, unsigned dim_pretrained_word,
  unsigned size_label, unsigned dim_label,
  unsigned dim_output,
  const std::unordered_map<unsigned, std::vector<float>>& pretrained):
  StaticInputLayer(model, size_word, dim_word, size_postag, dim_postag,
  size_pretrained_word, dim_pretrained_word, dim_output, pretrained),
  p_l(nullptr), p_l2l(nullptr), use_label(true) {
  if (dim_label == 0) {
    std::cerr << "Label embedding dim should be greater than 0." << std::endl;
    std::cerr << "Label embedding is inactivated." << std::endl;
    use_label = false;
  } else {
    p_l = model->add_lookup_parameters(size_label, { dim_label, 1 });
    p_l2l = model->add_parameters({ dim_output, dim_label });
  }
}


cnn::expr::Expression DynamicInputLayer::add_input2(cnn::ComputationGraph* hg,
  unsigned wid, unsigned pid, unsigned pre_wid, unsigned lid) {
  cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
  if (use_word && wid > 0) {
    cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
    cnn::expr::Expression w = cnn::expr::lookup(*hg, p_w, wid);
    expr = cnn::expr::affine_transform({ expr, w2l, w });
  }
  if (use_postag && pid > 0) {
    cnn::expr::Expression p2l = cnn::expr::parameter(*hg, p_p2l);
    cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
    expr = cnn::expr::affine_transform({ expr, p2l, p });
  }
  if (use_pretrained_word && pre_wid > 0) {
    cnn::expr::Expression t2l = cnn::expr::parameter(*hg, p_t2l);
    cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
    expr = cnn::expr::affine_transform({ expr, t2l, t });
  }
  if (use_label && lid > 0) {
    cnn::expr::Expression l2l = cnn::expr::parameter(*hg, p_l2l);
    cnn::expr::Expression l = cnn::expr::lookup(*hg, p_l, lid);
    expr = cnn::expr::affine_transform({ expr, l2l, l });
  }
  return cnn::expr::rectify(expr);
}


cnn::expr::Expression DynamicInputLayer::add_input2(cnn::ComputationGraph* hg,
  unsigned wid, unsigned pid, unsigned pre_wid, cnn::expr::Expression& lexpr) {
  cnn::expr::Expression expr = cnn::expr::parameter(*hg, p_ib);
  if (use_word && wid > 0) {
    cnn::expr::Expression w2l = cnn::expr::parameter(*hg, p_w2l);
    cnn::expr::Expression w = cnn::expr::lookup(*hg, p_w, wid);
    expr = cnn::expr::affine_transform({ expr, w2l, w });
  }
  if (use_postag && pid > 0) {
    cnn::expr::Expression p2l = cnn::expr::parameter(*hg, p_p2l);
    cnn::expr::Expression p = cnn::expr::lookup(*hg, p_p, pid);
    expr = cnn::expr::affine_transform({ expr, p2l, p });
  }
  if (use_pretrained_word && pre_wid > 0) {
    cnn::expr::Expression t2l = cnn::expr::parameter(*hg, p_t2l);
    cnn::expr::Expression t = cnn::expr::const_lookup(*hg, p_t, pre_wid);
    expr = cnn::expr::affine_transform({ expr, t2l, t });
  }
  if (use_label) {
    cnn::expr::Expression l2l = cnn::expr::parameter(*hg, p_l2l);
    expr = cnn::expr::affine_transform({ expr, l2l, lexpr });
  }
  return cnn::expr::rectify(expr);
}


LSTMLayer::LSTMLayer(cnn::Model* model,
  unsigned n_layers,
  unsigned dim_input,
  unsigned dim_hidden) : n_items(0),
  lstm(n_layers, dim_input, dim_hidden, model),
  p_guard(model->add_parameters({ dim_input, 1 })) {
}


void LSTMLayer::new_graph(cnn::ComputationGraph* hg) {
  lstm.new_graph(*hg);
}


void LSTMLayer::add_inputs(cnn::ComputationGraph* hg,
  const std::vector<cnn::expr::Expression>& inputs) {
  n_items = inputs.size();
  lstm.start_new_sequence();

  lstm.add_input(cnn::expr::parameter(*hg, p_guard));
  for (unsigned i = 0; i < n_items; ++i) {
    lstm.add_input(inputs[i]);
  }
}


cnn::expr::Expression LSTMLayer::get_output(cnn::ComputationGraph* hg, int index) {
  return lstm.get_h(cnn::RNNPointer(index + 1)).back();
}


void LSTMLayer::get_outputs(cnn::ComputationGraph* hg,
  std::vector<cnn::expr::Expression>& outputs) {
  outputs.resize(n_items);
  for (unsigned i = 0; i < n_items; ++i) {
    outputs[i] = get_output(hg, i);
  }
}


void LSTMLayer::set_dropout(float& rate) {
  lstm.set_dropout(rate);
}

void LSTMLayer::disable_dropout() {
  lstm.disable_dropout();
}

BidirectionalLSTMLayer::BidirectionalLSTMLayer(cnn::Model* model,
  unsigned n_lstm_layers,
  unsigned dim_lstm_input,
  unsigned dim_hidden) :
  n_items(0),
  fw_lstm(n_lstm_layers, dim_lstm_input, dim_hidden, model),
  bw_lstm(n_lstm_layers, dim_lstm_input, dim_hidden, model),
  p_fw_guard(model->add_parameters({ dim_lstm_input, 1 })),
  p_bw_guard(model->add_parameters({ dim_lstm_input, 1 })) {

}


void BidirectionalLSTMLayer::new_graph(cnn::ComputationGraph* hg) {
  fw_lstm.new_graph(*hg);
  bw_lstm.new_graph(*hg);
}


void BidirectionalLSTMLayer::add_inputs(cnn::ComputationGraph* hg,
  const std::vector<cnn::expr::Expression>& inputs) {
  n_items = inputs.size();
  fw_lstm.start_new_sequence();
  bw_lstm.start_new_sequence();

  fw_lstm.add_input(cnn::expr::parameter(*hg, p_fw_guard));
  for (unsigned i = 0; i < n_items; ++i) {
    fw_lstm.add_input(inputs[i]);
    bw_lstm.add_input(inputs[n_items - i - 1]);
  }
  bw_lstm.add_input(cnn::expr::parameter(*hg, p_bw_guard));
}


BidirectionalLSTMLayer::Output BidirectionalLSTMLayer::get_output(cnn::ComputationGraph* hg,
  int index) {
  return std::make_pair(
    fw_lstm.get_h(cnn::RNNPointer(index + 1)).back(),
    bw_lstm.get_h(cnn::RNNPointer(n_items - index - 1)).back());
}


void BidirectionalLSTMLayer::get_outputs(cnn::ComputationGraph* hg,
  std::vector<BidirectionalLSTMLayer::Output>& outputs) {
  outputs.resize(n_items);
  for (unsigned i = 0; i < n_items; ++i) {
    outputs[i] = get_output(hg, i);
  }
}


void BidirectionalLSTMLayer::set_dropout(float& rate) {
  fw_lstm.set_dropout(rate);
  bw_lstm.set_dropout(rate);
}


void BidirectionalLSTMLayer::disable_dropout() {
  fw_lstm.disable_dropout();
  bw_lstm.disable_dropout();
}


SoftmaxLayer::SoftmaxLayer(cnn::Model* model,
  unsigned dim_input,
  unsigned dim_output)
  : p_B(model->add_parameters({ dim_output, 1 })),
  p_W(model->add_parameters({ dim_output, dim_input })) {
}


cnn::expr::Expression SoftmaxLayer::get_output(cnn::ComputationGraph* hg,
  const cnn::expr::Expression& expr) {
  return cnn::expr::log_softmax(cnn::expr::affine_transform({
    cnn::expr::parameter(*hg, p_B), cnn::expr::parameter(*hg, p_W), expr }));
}


DenseLayer::DenseLayer(cnn::Model* model,
  unsigned dim_input,
  unsigned dim_output) :
  p_W(model->add_parameters({ dim_output, dim_input })),
  p_B(model->add_parameters({ dim_output, 1 })) {

}


cnn::expr::Expression DenseLayer::get_output(cnn::ComputationGraph* hg,
  const cnn::expr::Expression& expr) {
  return cnn::expr::affine_transform({
    cnn::expr::parameter(*hg, p_B),
    cnn::expr::parameter(*hg, p_W),
    expr });
}


Merge2Layer::Merge2Layer(cnn::Model* model,
  unsigned dim_input1,
  unsigned dim_input2,
  unsigned dim_output) : p_B(model->add_parameters({ dim_output, 1 })),
  p_W1(model->add_parameters({ dim_output, dim_input1 })),
  p_W2(model->add_parameters({ dim_output, dim_input2 })) {
}


cnn::expr::Expression Merge2Layer::get_output(cnn::ComputationGraph* hg,
  const cnn::expr::Expression& expr1,
  const cnn::expr::Expression& expr2) {
  cnn::expr::Expression i = cnn::expr::affine_transform({
    cnn::expr::parameter(*hg, p_B),
    cnn::expr::parameter(*hg, p_W1), expr1,
    cnn::expr::parameter(*hg, p_W2), expr2
  });
  return i;
}

Merge3Layer::Merge3Layer(cnn::Model* model,
  unsigned dim_input1,
  unsigned dim_input2,
  unsigned dim_input3,
  unsigned dim_output) : p_B(model->add_parameters({ dim_output, 1 })),
  p_W1(model->add_parameters({ dim_output, dim_input1 })),
  p_W2(model->add_parameters({ dim_output, dim_input2 })),
  p_W3(model->add_parameters({ dim_output, dim_input3 })) {
}


cnn::expr::Expression Merge3Layer::get_output(cnn::ComputationGraph* hg,
  const cnn::expr::Expression& expr1,
  const cnn::expr::Expression& expr2,
  const cnn::expr::Expression& expr3) {
  return cnn::expr::affine_transform({
    cnn::expr::parameter(*hg, p_B),
    cnn::expr::parameter(*hg, p_W1), expr1,
    cnn::expr::parameter(*hg, p_W2), expr2,
    cnn::expr::parameter(*hg, p_W3), expr3
  });
}


Merge4Layer::Merge4Layer(cnn::Model* model,
  unsigned dim_input1,
  unsigned dim_input2,
  unsigned dim_input3,
  unsigned dim_input4,
  unsigned dim_output) : p_B(model->add_parameters({ dim_output, 1 })),
  p_W1(model->add_parameters({ dim_output, dim_input1 })),
  p_W2(model->add_parameters({ dim_output, dim_input2 })),
  p_W3(model->add_parameters({ dim_output, dim_input3 })),
  p_W4(model->add_parameters({ dim_output, dim_input4 })) {
}


cnn::expr::Expression Merge4Layer::get_output(cnn::ComputationGraph* hg,
  const cnn::expr::Expression& expr1,
  const cnn::expr::Expression& expr2,
  const cnn::expr::Expression& expr3,
  const cnn::expr::Expression& expr4) {
  return cnn::expr::affine_transform({
    cnn::expr::parameter(*hg, p_B),
    cnn::expr::parameter(*hg, p_W1), expr1,
    cnn::expr::parameter(*hg, p_W2), expr2,
    cnn::expr::parameter(*hg, p_W3), expr3,
    cnn::expr::parameter(*hg, p_W4), expr4
  });
}


Merge5Layer::Merge5Layer(cnn::Model* model,
  unsigned dim_input1,
  unsigned dim_input2,
  unsigned dim_input3,
  unsigned dim_input4,
  unsigned dim_input5,
  unsigned dim_output) : p_B(model->add_parameters({ dim_output, 1 })),
  p_W1(model->add_parameters({ dim_output, dim_input1 })),
  p_W2(model->add_parameters({ dim_output, dim_input2 })),
  p_W3(model->add_parameters({ dim_output, dim_input3 })),
  p_W4(model->add_parameters({ dim_output, dim_input4 })),
  p_W5(model->add_parameters({ dim_output, dim_input5 })) {
}


cnn::expr::Expression Merge5Layer::get_output(cnn::ComputationGraph* hg,
  const cnn::expr::Expression& expr1,
  const cnn::expr::Expression& expr2,
  const cnn::expr::Expression& expr3,
  const cnn::expr::Expression& expr4,
  const cnn::expr::Expression& expr5) {
  return cnn::expr::affine_transform({
    cnn::expr::parameter(*hg, p_B),
    cnn::expr::parameter(*hg, p_W1), expr1,
    cnn::expr::parameter(*hg, p_W2), expr2,
    cnn::expr::parameter(*hg, p_W3), expr3,
    cnn::expr::parameter(*hg, p_W4), expr4,
    cnn::expr::parameter(*hg, p_W5), expr5
  });
}


Merge6Layer::Merge6Layer(cnn::Model* model,
  unsigned dim_input1,
  unsigned dim_input2,
  unsigned dim_input3,
  unsigned dim_input4,
  unsigned dim_input5,
  unsigned dim_input6,
  unsigned dim_output) : p_B(model->add_parameters({ dim_output, 1 })),
  p_W1(model->add_parameters({ dim_output, dim_input1 })),
  p_W2(model->add_parameters({ dim_output, dim_input2 })),
  p_W3(model->add_parameters({ dim_output, dim_input3 })),
  p_W4(model->add_parameters({ dim_output, dim_input4 })),
  p_W5(model->add_parameters({ dim_output, dim_input5 })), 
  p_W6(model->add_parameters({ dim_output, dim_input6 })) {
}


cnn::expr::Expression Merge6Layer::get_output(cnn::ComputationGraph* hg,
  const cnn::expr::Expression& expr1,
  const cnn::expr::Expression& expr2,
  const cnn::expr::Expression& expr3,
  const cnn::expr::Expression& expr4,
  const cnn::expr::Expression& expr5,
  const cnn::expr::Expression& expr6) {
  return cnn::expr::affine_transform({
    cnn::expr::parameter(*hg, p_B),
    cnn::expr::parameter(*hg, p_W1), expr1,
    cnn::expr::parameter(*hg, p_W2), expr2,
    cnn::expr::parameter(*hg, p_W3), expr3,
    cnn::expr::parameter(*hg, p_W4), expr4,
    cnn::expr::parameter(*hg, p_W5), expr5,
    cnn::expr::parameter(*hg, p_W6), expr6
  });
}


AttentionLayer::AttentionLayer(cnn::Model* model,
  unsigned size_word,
  unsigned size_windows,
  const std::set<unsigned>& vocab) : n_windows(size_windows), n_words(size_word) {
  p_K.resize(n_windows);
  for (unsigned i = 0; i < n_windows; ++i) {
    p_K[i] = model->add_lookup_parameters(size_word + n_windows, { 1, 1 });
    bool pivot = (i == n_windows / 2);
    std::vector<float> val(1); val[0] = (pivot ? 1.f : 1.f);
    for (unsigned j = 0; j < size_word; ++j) {
      if (0 == vocab.count(j)) { p_K[i]->Initialize(j, val); }
    }
  }
}


cnn::expr::Expression AttentionLayer::get_output(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& wids,
  const std::vector<cnn::expr::Expression>& inputs) {
  BOOST_ASSERT_MSG(wids.size() == n_windows, "# of input not match window size.");
  BOOST_ASSERT_MSG(inputs.size() == n_windows, "# of input not match window size.");
  std::vector<cnn::expr::Expression> weights(n_windows);
  for (unsigned i = 0; i < n_windows; ++i) {
    weights[i] = cnn::expr::lookup(*hg, p_K[i], wids[i]);
  }
  cnn::expr::Expression softmax = cnn::expr::softmax(cnn::expr::concatenate(weights));
  cnn::expr::Expression ret = cnn::expr::concatenate_cols(inputs) * softmax;
  return ret;
}


void StaticInputBidirectionalLSTM::get_inputs(cnn::ComputationGraph* hg,
  const std::vector<unsigned>& raw_sentence,
  const std::vector<unsigned>& sentence,
  const std::vector<std::string>& sentence_str,
  const std::vector<unsigned>& postags,
  std::vector<cnn::expr::Expression>& exprs) {

  unsigned len = sentence.size();
  exprs.resize(len);
  for (unsigned i = 0; i < len; ++i) {
    auto wid = sentence[i];
    auto pid = postags[i];
    auto pre_wid = raw_sentence[i];
    if (!pretrained.count(pre_wid)) { pre_wid = 0; }

    exprs[i] = input_layer.add_input(hg, wid, pid, pre_wid);
  }
}

