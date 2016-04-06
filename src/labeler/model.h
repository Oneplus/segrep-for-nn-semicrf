#ifndef __LSTM_MODEL_H__
#define __LSTM_MODEL_H__

#include "layer.h"


struct NERModel {
  const std::vector<std::string>& id_to_label;
  NERModel(const std::vector<std::string>& id_to_label);

  void get_possible_labels(std::vector<unsigned>& possible_labels);
  void get_possible_labels(unsigned prev, std::vector<unsigned>& possible_labels);
  unsigned get_best_scored_label(const std::vector<float>& scores,
    const std::vector<unsigned>& possible_labels);
};


struct SequenceLabelingModel {
  virtual void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels) = 0;

  virtual void explain(std::ostream& os,
    const std::map<unsigned, std::string>& id_to_word,
    const std::map<unsigned, std::string>& id_to_postag,
    const std::vector<std::string>& id_to_label) {}
};


struct LSTMNERLabeler : public SequenceLabelingModel, public NERModel {
  StaticInputLayer input_layer;
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  LSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label);

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct BiLSTMNERLabeler : public StaticInputBidirectionalLSTM, public SequenceLabelingModel, public NERModel {
  // [Input Layer]
  // X_i = relu(W_w W_i + W_p P_i + W_t T_i), where
  //  - W is tuned word embedding,
  //  - P is postag embedding,
  //  - T is the fix pretrained embedding.
  //
  // [Bidirectional LSTM Layer]
  // H_i = Bi-LSTM(X_[:i]), where
  //  - H_i is the hidden unit for LSTM cell
  //
  // [Softmax Layer]
  // Y_i = softmax(W_y H_i), where
  //  - Y_i is the final probability for each class.
  Merge2Layer merge2_layer;
  SoftmaxLayer softmax_layer;

  BiLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label):
    StaticInputBidirectionalLSTM(model, size_word, dim_word, size_postag, dim_postag,
    size_pretrained_word, dim_pretrained_word,
    n_lstm_layers, dim_lstm_input, dim_hidden,
    pretrained_embedding),
    NERModel(id_to_label),
    merge2_layer(model, dim_hidden, dim_hidden, dim_hidden),
    softmax_layer(model, dim_hidden, size_label) {
  }

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct TranBiLSTMNERLabeler : public StaticInputBidirectionalLSTM, public SequenceLabelingModel, public NERModel {
  Merge3Layer merge3_layer;
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;
  
  TranBiLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_label,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label);

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct GlobBiLSTMNERLabeler : public StaticInputBidirectionalLSTM, public SequenceLabelingModel, public NERModel {
  // 
  Merge4Layer merge4_layer;
  SoftmaxLayer softmax_layer;

  GlobBiLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label) :
    StaticInputBidirectionalLSTM(model, size_word, dim_word,
    size_postag, dim_postag,
    size_pretrained_word, dim_pretrained_word,
    n_lstm_layers, dim_lstm_input, dim_hidden,
    pretrained_embedding),
    NERModel(id_to_label),
    merge4_layer(model, dim_hidden, dim_hidden, dim_hidden, dim_hidden, dim_hidden),
    softmax_layer(model, dim_hidden, size_label) {
  }

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct LSTMBiLSTMNERLabeler : public StaticInputBidirectionalLSTM, public SequenceLabelingModel, public NERModel {
  // [Input Layer]
  // X_i = relu(W_w W_i + W_p P_i + W_t T_i), where
  //  - W is tuned word embedding,
  //  - P is postag embedding,
  //  - T is the fix pretrained embedding.
  //
  // [Bidirectional LSTM Layer]
  // H_i = Bi-LSTM(X_[:i]), where
  //  - H_i is the hidden unit for LSTM cell
  //
  // [LSTM Layer]
  // H'_i = LSTM(H_i), where
  //  - H'_i is the hidden unit for output LSTM cell
  // 
  // [Softmax Layer]
  // Y_i = softmax(W_y H'_i), where
  //  - Y_i is the final probability for each class.
  Merge2Layer merge2_layer;
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;

  LSTMBiLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label): 
    StaticInputBidirectionalLSTM(model, size_word, dim_word,
    size_postag, dim_postag,
    size_pretrained_word, dim_pretrained_word,
    n_lstm_layers, dim_lstm_input, dim_hidden,
    pretrained_embedding),
    NERModel(id_to_label),
    merge2_layer(model, dim_hidden, dim_hidden, dim_hidden),
    lstm_layer(model, n_lstm_layers, dim_hidden, dim_hidden),
    softmax_layer(model, dim_hidden, size_label) {
  }

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct LSTMAttendBiLSTMNERLabeler : public StaticInputBidirectionalLSTM,
  public SequenceLabelingModel, public NERModel {
  Merge2Layer merge2_layer;
  AttentionLayer attention_layer;
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;
  unsigned n_windows;
  unsigned n_words;
  std::vector<cnn::Parameters*> bos;
  std::vector<cnn::Parameters*> eos;
  
  LSTMAttendBiLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    unsigned size_attention_window,
    const std::set<unsigned>& vocab,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label):
    StaticInputBidirectionalLSTM(model, size_word, dim_word,
    size_postag, dim_postag,
    size_pretrained_word, dim_pretrained_word,
    n_lstm_layers, dim_lstm_input, dim_hidden, 
    pretrained_embedding),
    NERModel(id_to_label),
    merge2_layer(model, dim_hidden, dim_hidden, dim_hidden),
    attention_layer(model, size_word, size_attention_window, vocab),
    lstm_layer(model, n_lstm_layers, dim_hidden, dim_hidden),
    softmax_layer(model, dim_hidden, size_label),
    n_windows(size_attention_window),
    n_words(size_word),
    bos(), eos() {
    if (size_attention_window % 2 == 0) {
      std::cerr << "attention window size = " << size_attention_window << " is an even number.";
      std::cerr << " is it correct?" << std::endl;
    }
    bos.resize(size_attention_window / 2);
    eos.resize(size_attention_window / 2);
    for (unsigned i = 0; i < size_attention_window / 2; ++i) {
      bos[i] = model->add_parameters({ dim_hidden, 1 });
      eos[i] = model->add_parameters({ dim_hidden, 1 });
    }
  }

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);

  void explain(std::ostream& os,
    const std::map<unsigned, std::string>& id_to_word,
    const std::map<unsigned, std::string>& id_to_postag,
    const std::vector<std::string>& id_to_label);
};


struct Seq2SeqLabelVerBiLSTMNERLabeler : public StaticInputBidirectionalLSTM,
  public SequenceLabelingModel, public NERModel {
  Merge2Layer merge2_layer; // To merge two final embedding.
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;
  cnn::LookupParameters* p_l;
  cnn::Parameters* p_go;

  Seq2SeqLabelVerBiLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    unsigned dim_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label) : 
    StaticInputBidirectionalLSTM(model, size_word, dim_word,
    size_postag, dim_postag,
    size_pretrained_word, dim_pretrained_word,
    n_lstm_layers, dim_lstm_input, dim_hidden, 
    pretrained_embedding),
    NERModel(id_to_label),
    merge2_layer(model, dim_hidden, dim_hidden, dim_hidden),
    lstm_layer(model, n_lstm_layers, dim_label, dim_hidden),
    softmax_layer(model, dim_hidden, size_label),
    p_l(model->add_lookup_parameters(size_label, {dim_label, 1})),
    p_go(model->add_parameters({dim_label, 1})) {
  }

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct Seq2SeqWordVerBiLSTMNERLabeler : public SequenceLabelingModel, public NERModel {
  DynamicInputLayer dynamic_input_layer;
  cnn::Parameters *p_w2l0, *p_p2l0, *p_t2l0, *p_l0b;
  cnn::Parameters *p_guard;
  BidirectionalLSTMLayer bi_lstm_layer;
  Merge2Layer merge2_layer; // To merge two final embedding.
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  Seq2SeqWordVerBiLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    unsigned dim_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label);

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct Seq2SeqAttentionVerLSTMNERLabeler : public SequenceLabelingModel, public NERModel {
  StaticInputLayer static_input_layer;
  BidirectionalLSTMLayer bi_lstm_layer;
  cnn::LookupParameters *p_l;
  cnn::Parameters* p_guard;
  Merge2Layer merge2_layer;
  Merge3Layer merge3_layer; // To merge two final embedding.
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  Seq2SeqAttentionVerLSTMNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    unsigned dim_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label);

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


struct RevLSTM2SeqNERLabeler : public SequenceLabelingModel, public NERModel {
  // The LSTM network in http://www.aclweb.org/anthology/D/D15/D15-1042.pdf
  DynamicInputLayer input_layer;
  cnn::Parameters* p_guard;
  LSTMLayer lstm_layer;
  SoftmaxLayer softmax_layer;
  const std::unordered_map<unsigned, std::vector<float>>& pretrained;

  RevLSTM2SeqNERLabeler(cnn::Model* model,
    unsigned size_word,
    unsigned dim_word,
    unsigned size_postag,
    unsigned dim_postag,
    unsigned size_pretrained_word,
    unsigned dim_pretrained_word,
    unsigned dim_lstm_input,
    unsigned n_lstm_layers,
    unsigned dim_hidden,
    unsigned size_label,
    unsigned dim_label,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained_embedding,
    const std::vector<std::string>& id_to_label);

  void log_probability(cnn::ComputationGraph* hg,
    const std::vector<unsigned>& raw_sentence,
    const std::vector<unsigned>& sentence,
    const std::vector<std::string>& sentence_str,
    const std::vector<unsigned>& postags,
    const std::vector<unsigned>& correct_labels,
    double &n_correct,
    std::vector<unsigned>& predict_labels);
};


#endif  //  end for __LSTM_MODEL_H__
