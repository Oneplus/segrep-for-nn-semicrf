#ifndef __CRF_H__
#define __CRF_H__

#include "layer.h"
#include "cnn/cnn.h"

struct CRFBuilder {
  typedef std::vector<cnn::expr::Expression> ExpressionRow;

  StaticInputLayer input_layer;
  BidirectionalLSTMLayer bilstm_layer;
  Merge3Layer merge_layer;
  DenseLayer dense_layer;
  unsigned n_labels;
  float dropout_rate;
  cnn::LookupParameters* p_l;
  cnn::LookupParameters* p_tran;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  CRFBuilder(cnn::Model* model,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained_word, unsigned dim_pretrained_word,
    unsigned size_label, unsigned dim_label,
    unsigned n_layers, unsigned dim_lstm_input, unsigned dim_hidden,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding);

  cnn::expr::Expression supervised_loss(cnn::ComputationGraph* cg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<unsigned>& postag,
    const std::vector<unsigned>& correct);

  void viterbi(cnn::ComputationGraph* cg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<unsigned>& postag,
    std::vector<unsigned>& predict);
};

#endif  //  end for __CRF_H__