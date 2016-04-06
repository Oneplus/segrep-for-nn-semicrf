#include "semi_crf_layer.h"

SymbolEmbedding::SymbolEmbedding(cnn::Model& m, unsigned n, unsigned dim)
  :
  p_labels(m.add_lookup_parameters(n, { dim, 1 })) {
}

void SymbolEmbedding::new_graph(cnn::ComputationGraph& g) {
  cg = &g;
}

cnn::expr::Expression SymbolEmbedding::embed(unsigned label_id) {
  return cnn::expr::lookup(*cg, p_labels, label_id);
}

ConstSymbolEmbedding::ConstSymbolEmbedding(cnn::Model& m, unsigned n, unsigned dim)
  :
  p_labels(m.add_lookup_parameters(n, { dim, 1 })) {
}

void ConstSymbolEmbedding::new_graph(cnn::ComputationGraph& g) {
  cg = &g;
}

cnn::expr::Expression ConstSymbolEmbedding::embed(unsigned label_id) {
  return cnn::expr::const_lookup(*cg, p_labels, label_id);
}

MLPDurationEmbedding::MLPDurationEmbedding(cnn::Model& m, unsigned hidden, unsigned dim)
  :
  p_zero(m.add_parameters({ dim })),
  p_d2h(m.add_parameters({ hidden, dim })),
  p_hb(m.add_parameters({ hidden })),
  p_h2o(m.add_parameters({ dim, hidden })),
  p_ob(m.add_parameters({ dim })) {
  dur_xs.resize(10000, std::vector<float>(2));
  for (unsigned i = 1; i < dur_xs.size(); ++i) {
    dur_xs[i][0] = static_cast<float>(i);
    dur_xs[i][1] = static_cast<float>(i) / logf(2.f);
  }
}

void MLPDurationEmbedding::new_graph(cnn::ComputationGraph& g) {
  cg = &g;
  zero = cnn::expr::parameter(g, p_zero);
  d2h = cnn::expr::parameter(g, p_d2h);
  hb = cnn::expr::parameter(g, p_hb);
  h2o = cnn::expr::parameter(g, p_h2o);
  ob = cnn::expr::parameter(g, p_ob);
}

cnn::expr::Expression MLPDurationEmbedding::embed(unsigned dur) {
  if (dur) {
    cnn::expr::Expression x = cnn::expr::input(*cg, { 2 }, &dur_xs[dur]);
    cnn::expr::Expression h = cnn::expr::rectify(cnn::expr::affine_transform({ hb, d2h, x }));
    return cnn::expr::affine_transform({ ob, h2o, h });
  } else {
    return zero;
  }
}

BinnedDurationEmbedding::BinnedDurationEmbedding(cnn::Model& m, unsigned dim, unsigned n_bins)
  :
  p_e(m.add_lookup_parameters(n_bins, { dim, 1 })),
  max_bin(n_bins - 1) {
  BOOST_ASSERT(n_bins > 0);
}

void BinnedDurationEmbedding::new_graph(cnn::ComputationGraph& g) {
  cg = &g;
}

cnn::expr::Expression BinnedDurationEmbedding::embed(unsigned dur) {
  if (dur) {
    dur = static_cast<unsigned>(log(dur) / log(1.6f)) + 1;
  }
  if (dur > max_bin) {
    dur = max_bin;
  }
  return cnn::expr::lookup(*cg, p_e, dur);
}


SegUniEmbedding::SegUniEmbedding(cnn::Model& m, unsigned n_layers,
  unsigned lstm_input_dim, unsigned seg_dim)
  :
  p_h0(m.add_parameters({ lstm_input_dim })),
  builder(n_layers, lstm_input_dim, seg_dim, &m) {
}

void SegUniEmbedding::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  int max_seg_len) {
  len = c.size();
  h.clear(); // The first dimension for h is the starting point, the second is length.
  h.resize(len);
  cnn::expr::Expression h0 = cnn::expr::parameter(cg, p_h0);
  builder.new_graph(cg);
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    unsigned seg_len = max_j - i;
    auto& hi = h[i];
    hi.resize(seg_len);

    builder.start_new_sequence();
    builder.add_input(h0);
    // Put one span in h[i][j]
    for (unsigned k = 0; k < seg_len; ++k) {
      hi[k] = builder.add_input(c[i + k]);
    }
  }
}

const cnn::expr::Expression& SegUniEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j < len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}

void SegUniEmbedding::set_dropout(float& rate) {
  builder.set_dropout(rate);
}

void SegUniEmbedding::disable_dropout() {
  builder.disable_dropout();
}

SegBiEmbedding::SegBiEmbedding(cnn::Model& m, unsigned n_layers,
  unsigned lstm_input_dim, unsigned seg_dim)
  :
  fwd(m, n_layers, lstm_input_dim, seg_dim),
  bwd(m, n_layers, lstm_input_dim, seg_dim) {
}

void SegBiEmbedding::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c,
  int max_seg_len) {
  len = c.size();
  fwd.construct_chart(cg, c, max_seg_len);
  std::vector<cnn::expr::Expression> rc(len);
  for (unsigned i = 0; i < len; ++i) { rc[i] = c[len - i - 1]; }
  bwd.construct_chart(cg, rc, max_seg_len);
  h.clear();
  h.resize(len);
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    auto& hi = h[i];
    unsigned seg_len = max_j - i;
    hi.resize(seg_len);
    for (unsigned k = 0; k < seg_len; ++k) {
      unsigned j = i + k;
      const cnn::expr::Expression& fe = fwd(i, j);
      const cnn::expr::Expression& be = bwd(len - 1 - j, len - 1 - i);
      hi[k] = std::make_pair(fe, be);
    }
  }
}

const SegBiEmbedding::ExpressionPair& SegBiEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j < len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}

void SegBiEmbedding::set_dropout(float& rate) {
  fwd.set_dropout(rate);
  bwd.set_dropout(rate);
}

void SegBiEmbedding::disable_dropout() {
  fwd.disable_dropout();
  bwd.disable_dropout();
}

SegConvEmbedding::SegConvEmbedding(cnn::Model& m, unsigned input_dim,
  const std::vector<std::pair<unsigned, unsigned>>& info)
  : zeros(input_dim, 0.), filters_info(info) {
  unsigned n_filter_types = info.size();
  p_filters.resize(n_filter_types);
  p_biases.resize(n_filter_types);
  for (unsigned i = 0; i < info.size(); ++i) {
    const auto& filter_width = info[i].first;
    const auto& nb_filters = info[i].second;
    p_filters[i].resize(nb_filters);
    p_biases[i].resize(nb_filters);
    for (unsigned j = 0; j < nb_filters; ++j) {
      p_filters[i][j] = m.add_parameters({ input_dim, filter_width });
      p_biases[i][j] = m.add_parameters({ input_dim });
    }
  }
}

void SegConvEmbedding::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, int max_seg_len) {
  len = c.size();
  h.clear(); // The first dimension for h is the starting point, the second is length.
  h.resize(len);

  
  auto padding = cnn::expr::zeroes(cg, { zeros.size() });
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    unsigned seg_len = max_j - i;
    auto& hi = h[i];
    hi.resize(seg_len);

    std::vector<cnn::expr::Expression> s;
    for (unsigned k = 0; k < seg_len; ++k) {
      s.push_back(c[i + k]);

      std::vector<cnn::expr::Expression> tmp;
      for (unsigned ii = 0; ii < filters_info.size(); ++ii) {
        const auto& filter_width = filters_info[ii].first;
        const auto& nb_filters = filters_info[ii].second;

        for (unsigned p = 0; p < filter_width - 1; ++p) { s.push_back(padding); }
        for (unsigned jj = 0; jj < nb_filters; ++jj) {
          auto filter = cnn::expr::parameter(cg, p_filters[ii][jj]);
          auto bias = cnn::expr::parameter(cg, p_biases[ii][jj]);
          auto t = cnn::expr::conv1d_narrow(cnn::expr::concatenate_cols(s), filter);
          t = colwise_add(t, bias);
          t = cnn::expr::rectify(cnn::expr::kmax_pooling(t, 1));
          tmp.push_back(t);
        }
        for (unsigned p = 0; p < filter_width - 1; ++p) { s.pop_back(); }
      }
      hi[k] = cnn::expr::concatenate(tmp);
    }
  }
}

const cnn::expr::Expression& SegConvEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j < len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}

SegSimpleConcateEmbedding::SegSimpleConcateEmbedding(unsigned input_dim) : dim(input_dim) {

}

void SegSimpleConcateEmbedding::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, int max_seg_len) {
  len = c.size();
  h.clear(); // The first dimension for h is the starting point, the second is length.
  h.resize(len);

  cnn::expr::Expression z = cnn::expr::zeroes(cg, { dim });
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    unsigned seg_len = max_j - i;
    auto& hi = h[i];
    hi.resize(seg_len);

    std::vector<cnn::expr::Expression> s(max_seg_len, z);
    for (unsigned k = 0; k < seg_len; ++k) {
      s[k] = c[i + k];
      hi[k] = cnn::expr::concatenate(s);
    }
  }
}

const cnn::expr::Expression& SegSimpleConcateEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j < len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}


SegConcateEmbedding::SegConcateEmbedding(cnn::Model& m, unsigned input_dim, unsigned output_dim,
  unsigned max_seg_len) 
  : p_W(m.add_parameters({ output_dim, input_dim * max_seg_len })),
  p_b(m.add_parameters({output_dim})),
  paddings(input_dim, 0.) {
}

void SegConcateEmbedding::construct_chart(cnn::ComputationGraph& cg,
  const std::vector<cnn::expr::Expression>& c, int max_seg_len) {
  len = c.size();
  h.clear(); // The first dimension for h is the starting point, the second is length.
  h.resize(len);

  auto W = cnn::expr::parameter(cg, p_W);
  auto b = cnn::expr::parameter(cg, p_b);
  auto dim = p_W->dim.cols() / max_seg_len;
  auto z = cnn::expr::zeroes(cg, { dim });
  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    unsigned seg_len = max_j - i;
    auto& hi = h[i];
    hi.resize(seg_len);

    std::vector<cnn::expr::Expression> s(max_seg_len, z);
    for (unsigned k = 0; k < seg_len; ++k) {
      s[k] = c[i + k];
      hi[k] = cnn::expr::rectify(cnn::expr::affine_transform({ b, W, cnn::expr::concatenate(s) }));
    }
  }
}

const cnn::expr::Expression& SegConcateEmbedding::operator()(unsigned i, unsigned j) const {
  BOOST_ASSERT(j < len);
  BOOST_ASSERT(j >= i);
  return h[i][j - i];
}
