#ifndef __SEMI_CRF_LAYER_H__
#define __SEMI_CRF_LAYER_H__

#include "layer.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include <boost/functional/hash.hpp>

struct SymbolEmbedding {
  // Use to store the embedding for label.
  SymbolEmbedding(cnn::Model& m, unsigned n, unsigned dim);
  void new_graph(cnn::ComputationGraph& g);
  cnn::expr::Expression embed(unsigned label_id);

  cnn::ComputationGraph* cg;
  cnn::LookupParameters* p_labels;
};


struct ConstSymbolEmbedding {
  // Use to store the embedding for label.
  ConstSymbolEmbedding(cnn::Model& m, unsigned n, unsigned dim);
  void new_graph(cnn::ComputationGraph& g);
  cnn::expr::Expression embed(unsigned label_id);

  cnn::ComputationGraph* cg;
  cnn::LookupParameters* p_labels;
};


struct DurationEmbedding {
  virtual ~DurationEmbedding() {}
  virtual void new_graph(cnn::ComputationGraph& g) = 0;
  virtual cnn::expr::Expression embed(unsigned dur) = 0;
};


struct MLPDurationEmbedding : public DurationEmbedding {
  cnn::ComputationGraph* cg; //! The computation graph
  cnn::expr::Expression zero, d2h, hb, h2o, ob;
  std::vector<std::vector<float>> dur_xs; //! TODO
  cnn::Parameters* p_zero;  //! The zero parameters
  cnn::Parameters* p_d2h;   //! The input to hidden layer
  cnn::Parameters* p_hb;    //! The hidden layer bias
  cnn::Parameters* p_h2o;   //! The h2o layer
  cnn::Parameters* p_ob;    //! The output layer bias

  MLPDurationEmbedding(cnn::Model& m, unsigned hidden, unsigned dim);
  void new_graph(cnn::ComputationGraph& g) override;
  cnn::expr::Expression embed(unsigned dur) override;
};


struct BinnedDurationEmbedding : public DurationEmbedding {
  cnn::ComputationGraph* cg;
  cnn::LookupParameters* p_e;
  unsigned max_bin;

  BinnedDurationEmbedding(cnn::Model& m, unsigned hidden, unsigned n_bin = 8);
  void new_graph(cnn::ComputationGraph& g) override;
  cnn::expr::Expression embed(unsigned dur) override;
};


struct SegUniEmbedding {
  // uni-directional segment embedding.
  cnn::Parameters* p_h0;
  cnn::LSTMBuilder builder;
  std::vector<std::vector<cnn::expr::Expression>> h;
  unsigned len;

  explicit SegUniEmbedding(cnn::Model& m,
    unsigned n_layers, unsigned lstm_input_dim, unsigned seg_dim);
  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, int max_seg_len = 0);
  const cnn::expr::Expression& operator()(unsigned i, unsigned j) const;
  void set_dropout(float& rate);
  void disable_dropout();
};


struct SegBiEmbedding {
  typedef std::pair<cnn::expr::Expression, cnn::expr::Expression> ExpressionPair;
  SegUniEmbedding fwd, bwd;
  std::vector<std::vector<ExpressionPair>> h;
  unsigned len;

  explicit SegBiEmbedding(cnn::Model& m,
    unsigned n_layers, unsigned lstm_input_dim, unsigned seg_dim);
  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, int max_seg_len = 0);
  const ExpressionPair& operator()(unsigned i, unsigned j) const;
  void set_dropout(float& rate);
  void disable_dropout();
};


struct SegConvEmbedding {
  std::vector<std::vector<cnn::Parameters*>> p_filters;
  std::vector<std::vector<cnn::Parameters*>> p_biases;
  std::vector<float> zeros;
  std::vector<std::vector<cnn::expr::Expression>> h;
  std::vector<std::pair<unsigned, unsigned>> filters_info;
  unsigned len;

  explicit SegConvEmbedding(cnn::Model& m, unsigned input_dim, 
    const std::vector<std::pair<unsigned, unsigned>>& filters);
  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, int max_seg_len = 0);
  const cnn::expr::Expression& operator()(unsigned i, unsigned j) const;
};


struct SegSimpleConcateEmbedding {
  std::vector<std::vector<cnn::expr::Expression>> h;
  unsigned dim;
  unsigned len;

  explicit SegSimpleConcateEmbedding(unsigned input_dim);
  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, int max_seg_len = 0);
  const cnn::expr::Expression& operator()(unsigned i, unsigned j) const;
};


struct SegConcateEmbedding {
  cnn::Parameters* p_W;
  cnn::Parameters* p_b;
  std::vector<float> paddings;
  std::vector<std::vector<cnn::expr::Expression>> h;
  unsigned len;
  unsigned input_dim;
  unsigned max_seg_len;

  explicit SegConcateEmbedding(cnn::Model& m,
    unsigned input_dim,
    unsigned output_dim,
    unsigned max_seg_len);
  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, int max_seg_len = 0);
  const cnn::expr::Expression& operator()(unsigned i, unsigned j) const;
};


struct HashVector : public std::vector<unsigned> {
  bool operator == (const HashVector& other) const {
    if (size() != other.size()) { return false; }
    for (unsigned i = 0; i < size(); ++i) {
      if (at(i) != other.at(i)) { return false; }
    }
    return true;
  }
};

namespace std {
  template<>
  struct hash<HashVector> {
    std::size_t operator()(const HashVector& values) const {
      size_t seed = 0;
      boost::hash_range(seed, values.begin(), values.end());
      return seed;
    }
  };
}

#endif  //  end for __SEMI_CRF_LAYER_H__