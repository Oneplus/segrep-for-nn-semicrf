#ifndef __LAYER_H__
#define __LAYER_H__

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"

struct StaticInputLayer {
  cnn::LookupParameters* p_w;  // Word embedding
  cnn::LookupParameters* p_p;  // Postag embedding
  cnn::LookupParameters* p_t;  // Pretrained word embedding

  cnn::Parameters* p_ib;
  cnn::Parameters* p_w2l;
  cnn::Parameters* p_p2l;
  cnn::Parameters* p_t2l;

  bool use_word;
  bool use_postag;
  bool use_pretrained_word;

  StaticInputLayer(cnn::Model* model,
    unsigned size_word, unsigned dim_word,
    unsigned size_postag, unsigned dim_postag,
    unsigned size_pretrained_word, unsigned dim_pretrained_word,
    unsigned dim_output,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression add_input(cnn::ComputationGraph* hg,
    unsigned wid, unsigned pid, unsigned pre_wid);
};


struct DynamicInputLayer : public StaticInputLayer {
  cnn::LookupParameters* p_l;
  cnn::Parameters* p_l2l;
  bool use_label;

  DynamicInputLayer(cnn::Model* model,
    unsigned size_word, unsigned dim_word,
    unsigned size_postag, unsigned dim_postag,
    unsigned size_pretrained_word, unsigned dim_pretrained_word,
    unsigned size_label, unsigned dim_label,
    unsigned dim_output,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression add_input2(cnn::ComputationGraph* hg, 
    unsigned wid, unsigned pid, unsigned pre_wid, unsigned lid);

  cnn::expr::Expression add_input2(cnn::ComputationGraph* hg, 
    unsigned wid, unsigned pid, unsigned pre_wid, cnn::expr::Expression& expr);
};


struct LSTMLayer {
  unsigned n_items;
  cnn::LSTMBuilder lstm;
  cnn::Parameters* p_guard;

  LSTMLayer(cnn::Model* model, unsigned n_layers, unsigned dim_input, unsigned dim_hidden);
  void new_graph(cnn::ComputationGraph* hg);
  void add_inputs(cnn::ComputationGraph* hg, const std::vector<cnn::expr::Expression>& exprs);
  cnn::expr::Expression get_output(cnn::ComputationGraph* hg, int index);
  void get_outputs(cnn::ComputationGraph* hg, std::vector<cnn::expr::Expression>& outputs);
  void set_dropout(float& rate);
  void disable_dropout();
};


struct BidirectionalLSTMLayer {
  typedef std::pair<cnn::expr::Expression, cnn::expr::Expression> Output;
  unsigned n_items;
  cnn::LSTMBuilder fw_lstm;
  cnn::LSTMBuilder bw_lstm;
  cnn::Parameters* p_fw_guard;
  cnn::Parameters* p_bw_guard;

  BidirectionalLSTMLayer(cnn::Model* model,
    unsigned n_lstm_layers,
    unsigned dim_lstm_input,
    unsigned dim_hidden);

  void new_graph(cnn::ComputationGraph* hg);
  void add_inputs(cnn::ComputationGraph* hg, const std::vector<cnn::expr::Expression>& exprs);
  Output get_output(cnn::ComputationGraph* hg, int index);
  void get_outputs(cnn::ComputationGraph* hg, std::vector<Output>& outputs);
  void set_dropout(float& rate);
  void disable_dropout();
};


struct SoftmaxLayer {
  cnn::Parameters* p_B;
  cnn::Parameters* p_W;

  SoftmaxLayer(cnn::Model* model, unsigned dim_input, unsigned dim_output);
  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr);
};


struct DenseLayer {
  cnn::Parameters *p_W, *p_B;
  DenseLayer(cnn::Model* model, unsigned dim_input, unsigned dim_output);
  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr);
};


struct Merge2Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2;

  Merge2Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2);
};


struct Merge3Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2, *p_W3;

  Merge3Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3);
};


struct Merge4Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2, *p_W3, *p_W4;

  Merge4Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_input4,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3,
    const cnn::expr::Expression& expr4);
};


struct Merge5Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2, *p_W3, *p_W4, *p_W5;

  Merge5Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_input4,
    unsigned dim_input5,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3,
    const cnn::expr::Expression& expr4,
    const cnn::expr::Expression& expr5);
};


struct Merge6Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2, *p_W3, *p_W4, *p_W5, *p_W6;

  Merge6Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_input4,
    unsigned dim_input5,
    unsigned dim_input6,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3,
    const cnn::expr::Expression& expr4,
    const cnn::expr::Expression& expr5,
    const cnn::expr::Expression& expr6);
};


struct AttentionLayer {
  unsigned n_windows;
  unsigned n_words;
  std::vector<cnn::LookupParameters*> p_K;
  // const std::set<unsigned>& training_vocab;

  AttentionLayer(cnn::Model* model, unsigned size_word, unsigned n_windows,
    const std::set<unsigned>& vocab);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& wids,
    const std::vector<cnn::expr::Expression>& inputs);
};


struct StaticInputBidirectionalLSTM {
  StaticInputLayer input_layer;
  BidirectionalLSTMLayer bi_lstm_layer;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  StaticInputBidirectionalLSTM(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned n_lstm_layers,
    unsigned dim_lstm_input,
    unsigned dim_lstm_output,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding) :
    input_layer(model, size_word, dim_word,
    size_postag, dim_postag,
    size_pretrained_word, dim_pretrained_word,
    dim_lstm_input, pretrained_embedding),
    bi_lstm_layer(model, n_lstm_layers, dim_lstm_input, dim_lstm_output),
    pretrained(pretrained_embedding) {
  }

  void get_inputs(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    std::vector<cnn::expr::Expression>& exprs);
};

#endif  //  end for __LAYER_H__
