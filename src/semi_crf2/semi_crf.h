#ifndef __SEMI_CRF_H__
#define __SEMI_CRF_H__

#include "layer.h"
#include "semi_crf_layer.h"
#include "corpus.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include <unordered_map>


struct SemiCRFBuilder {
  virtual cnn::expr::Expression supervised_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0) = 0;

  virtual void viterbi(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    Corpus::Segmentation& yz_pred,
    unsigned max_seg_len = 0) = 0;
};


struct ZerothOrderSemiCRFBuilder : public SemiCRFBuilder {
  StaticInputLayer input_layer;
  BidirectionalLSTMLayer bilstm_layer;
  Merge2Layer merge2;
  BinnedDurationEmbedding dur_emb;
  float dropout_rate;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  ZerothOrderSemiCRFBuilder(cnn::Model& m,
    unsigned size_char, unsigned dim_char,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_duration,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression supervised_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentencne,
    const Corpus::Sentence& sentence,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0) override;

  void viterbi(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    Corpus::Segmentation& yz_pred,
    unsigned max_seg_len = 0) override;

  virtual void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) = 0;

  virtual cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, bool train) = 0;
};


struct ZerothOrderRnnSemiCRFBuilder : public ZerothOrderSemiCRFBuilder {
  SegBiEmbedding seg_emb;
  Merge3Layer merge3;
  DenseLayer dense;

  explicit ZerothOrderRnnSemiCRFBuilder(cnn::Model& m,
    unsigned size_char, unsigned dim_char,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_hidden2,
    unsigned dim_duration,
    unsigned dim_seg,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train);

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, bool train);
};


struct ZerothOrderConcateSemiCRFBuilder : public ZerothOrderSemiCRFBuilder {
  SegConcateEmbedding seg_emb;
  Merge2Layer merge2;
  DenseLayer dense;

  explicit ZerothOrderConcateSemiCRFBuilder(cnn::Model& m,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train);

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, bool train);
};

struct ZerothOrderCnnSemiCRFBuilder : public ZerothOrderSemiCRFBuilder {
  SegConvEmbedding seg_emb;
  Merge2Layer merge2;
  DenseLayer dense;

  explicit ZerothOrderCnnSemiCRFBuilder(cnn::Model& m,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};

struct ZerothOrderSimpleConcateSemiCRFBuilder : public ZerothOrderSemiCRFBuilder {
  SegSimpleConcateEmbedding seg_emb;
  Merge2Layer merge2;
  DenseLayer dense;

  explicit ZerothOrderSimpleConcateSemiCRFBuilder(cnn::Model& m,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train);

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, bool train);
};


struct FirstOrderSemiCRFBuilder : public SemiCRFBuilder {
  StaticInputLayer input_layer;
  BidirectionalLSTMLayer bilstm_layer;
  Merge2Layer merge2;
  BinnedDurationEmbedding dur_emb;
  float dropout_rate;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  explicit FirstOrderSemiCRFBuilder(cnn::Model& m,
    unsigned size_char, unsigned dim_char,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_duration,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression supervised_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0);

  void viterbi(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    Corpus::Segmentation& yz_pred,
    unsigned max_seg_len = 0);

  virtual void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) = 0;

  virtual cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, unsigned i, bool train) = 0;
  
  virtual cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, bool train) = 0;
};


struct FirstOrderRnnSemiCRFBuilder : public FirstOrderSemiCRFBuilder {
  SegBiEmbedding seg_emb;
  Merge6Layer merge6;
  DenseLayer dense;
  cnn::Parameters *p_fw_seg_guard, *p_bw_seg_guard, *p_dur_guard;

  explicit FirstOrderRnnSemiCRFBuilder(cnn::Model& m,
    unsigned size_char, unsigned dim_char,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_hidden2,
    unsigned dim_duration,
    unsigned dim_seg,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
  
  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, unsigned i, bool train) override;
  
  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, bool train) override;
};

struct FirstOrderConcateSemiCRFBuilder : public FirstOrderSemiCRFBuilder {
  SegConcateEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;
  cnn::Parameters *p_seg_guard, *p_dur_guard;

  explicit FirstOrderConcateSemiCRFBuilder(cnn::Model& m,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, unsigned i, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, bool train) override;
};

struct FirstOrderSimpleConcateSemiCRFBuilder : public FirstOrderSemiCRFBuilder {
  SegSimpleConcateEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;
  cnn::Parameters *p_seg_guard, *p_dur_guard;

  explicit FirstOrderSimpleConcateSemiCRFBuilder(cnn::Model& m,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, unsigned i, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned k, unsigned j, bool train) override;
};


struct ZerothOrderSemiCRFwLexBuilder : public SemiCRFBuilder {
  StaticInputLayer input_layer;
  BidirectionalLSTMLayer bilstm_layer;
  Merge2Layer merge2;
  BinnedDurationEmbedding dur_emb;
  float dropout_rate;
  const std::unordered_map<HashVector, unsigned>& lexicon;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  ZerothOrderSemiCRFwLexBuilder(cnn::Model& m,
    unsigned size_char, unsigned dim_char,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_duration,
    float dropout_rate,
    const std::unordered_map<HashVector, unsigned>& lexicon,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression supervised_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentencne,
    const Corpus::Sentence& sentence,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0) override;

  void viterbi(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    Corpus::Segmentation& yz_pred,
    unsigned max_seg_len = 0) override;

  unsigned get_lexicon_feature(const Corpus::Sentence& raw_sentence,
    unsigned i, unsigned j);

  virtual void new_graph(cnn::ComputationGraph& cg) = 0;

  virtual void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) = 0;

  virtual cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned l, bool train) = 0;
};


struct ZerothOrderRnnSemiCRFwRichLexBuilder : public ZerothOrderSemiCRFwLexBuilder {
  SegBiEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;
  ConstSymbolEmbedding lex_emb;

  explicit ZerothOrderRnnSemiCRFwRichLexBuilder(cnn::Model& m,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void new_graph(cnn::ComputationGraph& cg);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned l, bool train) override;
};


struct ZerothOrderConcateSemiCRFwRichLexBuilder : public ZerothOrderSemiCRFwLexBuilder {
  SegConcateEmbedding seg_emb;
  Merge3Layer merge3;
  DenseLayer dense;
  ConstSymbolEmbedding lex_emb;

  explicit ZerothOrderConcateSemiCRFwRichLexBuilder(cnn::Model& m,
    unsigned size_char, unsigned dim_char,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_lex, unsigned dim_lex,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned max_seg_len,
    unsigned dim_seg, // should be equal to dim_hidden1 * max_seg_len
    unsigned dim_duration,
    unsigned dim_hidden2,
    float dropout_rate,
    const std::unordered_map<HashVector, unsigned>& lexicon,
    std::vector<std::vector<float>>& entities_embedding,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void new_graph(cnn::ComputationGraph& cg);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned l, bool train) override;
};


struct ZerothOrderRnnSemiCRFwTuneRichLexBuilder : public ZerothOrderSemiCRFwLexBuilder {
  SegBiEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;
  SymbolEmbedding lex_emb;

  explicit ZerothOrderRnnSemiCRFwTuneRichLexBuilder(cnn::Model& m,
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
    bool init_segment_embedding);

  void new_graph(cnn::ComputationGraph& cg);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned l, bool train) override;
};


struct ZerothOrderConcateSemiCRFwTuneRichLexBuilder : public ZerothOrderSemiCRFwLexBuilder {
  SegConcateEmbedding seg_emb;
  Merge3Layer merge3;
  DenseLayer dense;
  SymbolEmbedding lex_emb;

  explicit ZerothOrderConcateSemiCRFwTuneRichLexBuilder(cnn::Model& m,
    unsigned size_char, unsigned dim_char,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_lex, unsigned dim_lex,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned max_seg_len,
    unsigned dim_seg, // should be equal to dim_hidden1 * max_seg_len
    unsigned dim_duration,
    unsigned dim_hidden2,
    float dropout_rate,
    const std::unordered_map<HashVector, unsigned>& lexicon,
    std::vector<std::vector<float>>& entities_embedding,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained,
    bool init_segment_embedding);

  void new_graph(cnn::ComputationGraph& cg);

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned l, bool train) override;
};


#endif  //  end for __SEMI_CRF_H__
