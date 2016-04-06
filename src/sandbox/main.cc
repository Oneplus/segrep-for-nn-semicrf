#include <iostream>
#include <vector>
#include "cnn/cnn.h"
#include "cnn/expr.h"

void test1() {
  const unsigned n_tokens = 5;
  const unsigned dim_token = 3;
  const unsigned dim_output = 2;
  std::vector<std::vector<float>> in(5);
  float cursor = 0.;
  for (unsigned i = 0; i < n_tokens; ++i) {
    in[i].resize(dim_token);
    for (unsigned j = 0; j < dim_token; ++j) { in[i][j] = cursor; cursor += 1; }
  }
  std::vector<cnn::expr::Expression> inputs(n_tokens);
  cnn::ComputationGraph cg;
  cnn::Model m;
  for (unsigned i = 0; i < n_tokens; ++i) {
    inputs[i] = cnn::expr::input(cg, { dim_token }, in[i]);
  }
  //cnn::Parameters* p = m.add_parameters({dim_output, dim_token}, 0.);
  cnn::Parameters* p = m.add_parameters({ dim_token, dim_output }, 0.);
  cnn::expr::Expression pack = cnn::expr::concatenate_cols(inputs);
  cnn::expr::Expression emb = cnn::expr::conv1d_narrow(pack, cnn::expr::parameter(cg, p));
  cg.PrintGraphviz();
}

void test2() {
  const unsigned n_tokens = 1;
  const unsigned dim_token = 3;
  const unsigned filter_width = 2;
  const unsigned dim_output = 4;
  std::vector<std::vector<float>> in(5);
  float cursor = 0.;
  for (unsigned i = 0; i < n_tokens; ++i) {
    in[i].resize(dim_token);
    for (unsigned j = 0; j < dim_token; ++j) { in[i][j] = cursor; cursor += 1; }
  }
  std::vector<cnn::expr::Expression> inputs(n_tokens);
  cnn::ComputationGraph cg;
  cnn::Model m;
  for (unsigned i = 0; i < n_tokens; ++i) {
    inputs[i] = cnn::expr::input(cg, { dim_token }, in[i]);
  }
  //cnn::Parameters* p = m.add_parameters({dim_output, dim_token}, 0.);
  cnn::Parameters* p = m.add_parameters({ dim_token, filter_width }, 0.);
  cnn::expr::Expression pack = cnn::expr::concatenate_cols(inputs);
  cnn::expr::Expression conv = cnn::expr::conv1d_wide(pack, cnn::expr::parameter(cg, p));
  cnn::expr::Expression embed = cnn::expr::kmax_pooling(conv, 1);
  std::vector<float> val = cnn::as_vector(cg.get_value(embed));
  for (auto& v : val) { std::cerr << v << std::endl; }
  cg.PrintGraphviz();
}

int main(int argc, char* argv[]) {
  cnn::Initialize(argc, argv, 1234);
  test2();
  return 0;
}