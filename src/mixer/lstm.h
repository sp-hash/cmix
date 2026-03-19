#ifndef LSTM_COMPRESS_H
#define LSTM_COMPRESS_H

#include <valarray>
#include <vector>
#include <memory>
#include <string>

#include "lstm-layer.h"

class Lstm {
 public:
  Lstm(unsigned int input_size, unsigned int output_size, unsigned int
      num_cells, unsigned int num_layers, int horizon, float learning_rate,
      float gradient_clip);
  ~Lstm();
  std::valarray<float>& Perceive(unsigned int input);
  std::valarray<float>& Predict(unsigned int input);
  void SetInput(const std::valarray<float>& input);
  void SaveToDisk(const std::string& path);
  void LoadFromDisk(const std::string& path);

 private:
  std::vector<std::unique_ptr<LstmLayer>> layers_;
  std::vector<unsigned int> input_history_;
  std::valarray<float> hidden_, hidden_error_;
  // Flattened buffers for better locality: layout per-epoch
  // layer_input_flat_: concatenation of all layers' input vectors for each epoch
  std::vector<float> layer_input_flat_;
  std::vector<unsigned int> layer_input_layer_offset_; // per-layer offset inside an epoch
  std::vector<unsigned int> layer_input_size_per_layer_;
  std::vector<float> output_layer_flat_; // [epoch][output_i][hidden]
  std::vector<float> output_flat_; // [epoch][output_i]
  std::valarray<float> outputs_single_;
  float learning_rate_;
  unsigned int num_cells_, epoch_, horizon_, input_size_, output_size_;
  unsigned int num_layers_;
};

#endif

