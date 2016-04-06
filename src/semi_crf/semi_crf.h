#ifndef __SEMI_CRF_H__
#define __SEMI_CRF_H__

#include "layer.h"
#include "semi_crf_layer.h"
#include "corpus.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include <unordered_map>


struct SemiCRFBuilderI {
  virtual cnn::expr::Expression supervised_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0) = 0;

  virtual cnn::expr::Expression margin_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0) = 0;

  virtual void viterbi(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    Corpus::Segmentation& yz_pred,
    unsigned max_seg_len = 0) = 0;
};

struct SemiCRFBuilder : public SemiCRFBuilderI {
  StaticInputLayer input_layer;
  BidirectionalLSTMLayer bilstm_layer;
  Merge2Layer merge2;
  SymbolEmbedding y_emb;
  BinnedDurationEmbedding dur_emb;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;
  unsigned size_tag;
  float dropout_rate;

  SemiCRFBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_duration,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression supervised_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0);

  cnn::expr::Expression margin_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0);

  void viterbi(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    Corpus::Segmentation& yz_pred,
    unsigned max_seg_len = 0);

  virtual cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, bool train) = 0;

  virtual void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) = 0;
};


struct RNNSemiCRFBuilder: public SemiCRFBuilder {
  SegBiEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;

  explicit RNNSemiCRFBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_duration,
    unsigned dim_seg,
    unsigned dim_hidden2,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};


struct CNNSemiCRFBuilder : public SemiCRFBuilder {
  SegConvEmbedding seg_emb;
  Merge3Layer merge3;
  DenseLayer dense;

  explicit CNNSemiCRFBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
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
    unsigned i, unsigned j, unsigned tag, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};

struct ConcateSemiCRFBuilder : public SemiCRFBuilder {
  SegConcateEmbedding seg_emb;
  Merge3Layer merge3;
  DenseLayer dense;

  explicit ConcateSemiCRFBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned max_seg_len,
    unsigned dim_seg,
    unsigned dim_duration,
    unsigned dim_hidden2,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};


struct SimpleConcateSemiCRFBuilder : public SemiCRFBuilder {
  SegSimpleConcateEmbedding seg_emb;
  Merge3Layer merge3;
  DenseLayer dense;

  explicit SimpleConcateSemiCRFBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned max_seg_len,
    unsigned dim_seg,
    unsigned dim_duration,
    unsigned dim_hidden2,
    float dropout_rate,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};


struct SemiCRFwLexBuilder : public SemiCRFBuilderI {
  StaticInputLayer input_layer;
  BidirectionalLSTMLayer bilstm_layer;
  Merge2Layer merge2;
  SymbolEmbedding y_emb;
  BinnedDurationEmbedding dur_emb;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;
  const std::unordered_map<HashVector, unsigned>& lexicon;
  unsigned size_tag;
  float dropout_rate;
  
  SemiCRFwLexBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_duration,
    float dropout_rate,
    const std::unordered_map<HashVector, unsigned>& lexicon,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression supervised_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0);

  cnn::expr::Expression margin_loss(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    const Corpus::Segmentation& correct,
    unsigned max_seg_len = 0);

  void viterbi(cnn::ComputationGraph& cg,
    const Corpus::Sentence& raw_sentence,
    const Corpus::Sentence& sentence,
    const Corpus::Sentence& postag,
    Corpus::Segmentation& yz_pred,
    unsigned max_seg_len = 0);
  
  unsigned get_lexicon_feature(const Corpus::Sentence& raw_sentence,
    unsigned i, unsigned j);

  virtual void new_graph(cnn::ComputationGraph& cg) = 0;

  virtual cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, unsigned lex, bool train) = 0;

  virtual void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) = 0;
};


struct RNNSemiCRFwLexBuilder : public SemiCRFwLexBuilder {
  SegBiEmbedding seg_emb;
  Merge5Layer merge5;
  DenseLayer dense;
  SymbolEmbedding lex_emb;

  explicit RNNSemiCRFwLexBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
    unsigned size_lex, unsigned dim_lex,
    unsigned n_layers,
    unsigned lstm_input_dim,
    unsigned dim_hidden1,
    unsigned dim_duration,
    unsigned dim_seg,
    unsigned dim_hidden2,
    float dropout_rate,
    const std::unordered_map<HashVector, unsigned>& lexicon,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void new_graph(cnn::ComputationGraph& cg) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, unsigned lex, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};


struct ConcateSemiCRFwLexBuilder : public SemiCRFwLexBuilder {
  SegConcateEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;
  SymbolEmbedding lex_emb;

  explicit ConcateSemiCRFwLexBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void new_graph(cnn::ComputationGraph& cg) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, unsigned lex, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};


struct RNNSemiCRFwRichLexBuilder : public SemiCRFwLexBuilder {
  SegBiEmbedding seg_emb;
  Merge5Layer merge5;
  DenseLayer dense;
  ConstSymbolEmbedding lex_emb;

  explicit RNNSemiCRFwRichLexBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void new_graph(cnn::ComputationGraph& cg) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, unsigned lex, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};

struct ConcateSemiCRFwRichLexBuilder : public SemiCRFwLexBuilder {
  SegConcateEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;
  ConstSymbolEmbedding lex_emb;

  explicit ConcateSemiCRFwRichLexBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  void new_graph(cnn::ComputationGraph& cg) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, unsigned lex, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};


struct RNNSemiCRFwTuneRichLexBuilder : public SemiCRFwLexBuilder {
  SegBiEmbedding seg_emb;
  Merge5Layer merge5;
  DenseLayer dense;
  SymbolEmbedding lex_emb;

  explicit RNNSemiCRFwTuneRichLexBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained,
    bool init_segment_embed);

  void new_graph(cnn::ComputationGraph& cg) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, unsigned lex, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};

struct ConcateSemiCRFwTuneRichLexBuilder : public SemiCRFwLexBuilder {
  SegConcateEmbedding seg_emb;
  Merge4Layer merge4;
  DenseLayer dense;
  SymbolEmbedding lex_emb;

  explicit ConcateSemiCRFwTuneRichLexBuilder(cnn::Model& m,
    unsigned size_word, unsigned dim_word,
    unsigned size_pos, unsigned dim_pos,
    unsigned size_pretrained, unsigned dim_pretrained,
    unsigned size_tag, unsigned dim_tag,
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
    const std::unordered_map<unsigned, std::vector<float>>& pretrained,
    bool init_segment_embed);

  void new_graph(cnn::ComputationGraph& cg) override;

  cnn::expr::Expression factor_score(cnn::ComputationGraph& cg,
    unsigned i, unsigned j, unsigned tag, unsigned lex, bool train) override;

  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, unsigned max_seg_len, bool train) override;
};

#endif  //  end for __SEMI_CRF_H__
