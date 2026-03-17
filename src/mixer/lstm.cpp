#include "lstm.h"

#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <iostream>

// small SIMD helpers (AVX-512 or AVX2 if available, scalar fallback)
#include <immintrin.h>

static inline float dot_product_f(const float* a, const float* b, int n) {
  int i = 0;
  float sum = 0;
#ifdef __AVX512F__
  __m512 vsum = _mm512_setzero_ps();
  for (; i + 15 < n; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    vsum = _mm512_add_ps(vsum, _mm512_mul_ps(va, vb));
  }
  sum = _mm512_reduce_add_ps(vsum);
#elif defined(__AVX2__)
  __m256 vsum2 = _mm256_setzero_ps();
  for (; i + 7 < n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    vsum2 = _mm256_add_ps(vsum2, _mm256_mul_ps(va, vb));
  }
  alignas(32) float tmp[8];
  _mm256_storeu_ps(tmp, vsum2);
  sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#endif
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

static inline void saxpy_f(float* y, const float* x, int n, float a) {
  int i = 0;
#ifdef __AVX512F__
  __m512 va = _mm512_set1_ps(a);
  for (; i + 15 < n; i += 16) {
    __m512 vx = _mm512_loadu_ps(x + i);
    __m512 vy = _mm512_loadu_ps(y + i);
    _mm512_storeu_ps(y + i, _mm512_add_ps(vy, _mm512_mul_ps(va, vx)));
  }
#elif defined(__AVX2__)
  __m256 va2 = _mm256_set1_ps(a);
  for (; i + 7 < n; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    _mm256_storeu_ps(y + i, _mm256_add_ps(vy, _mm256_mul_ps(va2, vx)));
  }
#endif
  for (; i < n; ++i) y[i] += a * x[i];
}

Lstm::Lstm(unsigned int input_size, unsigned int output_size, unsigned int
    num_cells, unsigned int num_layers, int horizon, float learning_rate,
    float gradient_clip) : input_history_(horizon),
    hidden_(num_cells * num_layers + 1), hidden_error_(num_cells),
    layer_input_(std::valarray<std::valarray<float>>(std::valarray<float>
    (input_size + 1 + num_cells * 2), num_layers), horizon),
    output_layer_(std::valarray<std::valarray<float>>(std::valarray<float>
    (num_cells * num_layers + 1), output_size), horizon),
    output_(std::valarray<float>(1.0 / output_size, output_size), horizon),
    learning_rate_(learning_rate), num_cells_(num_cells), epoch_(0),
    horizon_(horizon), input_size_(input_size), output_size_(output_size) {
  hidden_[hidden_.size() - 1] = 1;
  for (int epoch = 0; epoch < horizon; ++epoch) {
    layer_input_[epoch][0].resize(1 + num_cells + input_size);
    for (unsigned int i = 0; i < num_layers; ++i) {
      layer_input_[epoch][i][layer_input_[epoch][i].size() - 1] = 1;
    }
  }
  for (unsigned int i = 0; i < num_layers; ++i) {
    layers_.push_back(std::unique_ptr<LstmLayer>(new LstmLayer(
        layer_input_[0][i].size() + output_size, input_size_, output_size_,
        num_cells, horizon, gradient_clip, learning_rate)));
  }
  //LoadFromDisk("lstm.dat");
}

Lstm::~Lstm() {
  //SaveToDisk("lstm.dat");
}

void Lstm::SaveToDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ofstream os(path, std::ios::binary | std::ios::out);
  if (!os.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    os.write(reinterpret_cast<const char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i]->Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        os.write(reinterpret_cast<const char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  os.close();
}

void Lstm::LoadFromDisk(const std::string& path) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  std::ifstream is(path, std::ios::binary | std::ios::in);
  if (!is.is_open()) return;
  for (int i = 0; i < output_size_; ++i) {
    is.read(reinterpret_cast<char*>(&output_layer_[last_epoch][i][0]),
        std::streamsize(output_layer_[0][i].size() * sizeof(float)));
  }
  for (int i = 0; i < layers_.size(); ++i) {
    auto weights = layers_[i]->Weights();
    for (int j = 0; j < weights.size(); ++j) {
      for (int k = 0; k < weights[j]->size(); ++k) {
        is.read(reinterpret_cast<char*>(&(*weights[j])[k][0]),
          std::streamsize((*weights[j])[k].size() * sizeof(float)));
      }
    }
  }
  is.close();
}

void Lstm::SetInput(const std::valarray<float>& input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    std::copy(begin(input), begin(input) + input_size_,
        begin(layer_input_[epoch_][i]));
  }
}

std::valarray<float>& Lstm::Perceive(unsigned int input) {
  int last_epoch = epoch_ - 1;
  if (last_epoch == -1) last_epoch = horizon_ - 1;
  int old_input = input_history_[last_epoch];
  input_history_[last_epoch] = input;
  if (epoch_ == 0) {
    for (int epoch = horizon_ - 1; epoch >= 0; --epoch) {
      for (int layer = layers_.size() - 1; layer >= 0; --layer) {
        int offset = layer * num_cells_;
        // accumulate hidden_error_ += output_layer_[epoch][i][offset..] * error
        for (unsigned int i = 0; i < output_size_; ++i) {
          float error = (i == input_history_[epoch]) ? (output_[epoch][i] - 1) : output_[epoch][i];
          const float* out_ptr = &output_layer_[epoch][i][0] + offset;
          float* herr_ptr = &hidden_error_[0];
          saxpy_f(herr_ptr, out_ptr, hidden_error_.size(), error);
        }
        int prev_epoch = epoch - 1;
        if (prev_epoch == -1) prev_epoch = horizon_ - 1;
        int input_symbol = input_history_[prev_epoch];
        if (epoch == 0) input_symbol = old_input;
        layers_[layer]->BackwardPass(layer_input_[epoch][layer], epoch, layer,
            input_symbol, &hidden_error_);
      }
    }
  }

  for (unsigned int i = 0; i < output_size_; ++i) {
    float error = (i == input) ? (output_[last_epoch][i] - 1) : output_[last_epoch][i];
    float* dest = &output_layer_[epoch_][i][0];
    const float* src = &output_layer_[last_epoch][i][0];
    const float* hid = &hidden_[0];
    float alpha = -learning_rate_ * error;
    int n = hidden_.size();

    int j = 0;
#ifdef __AVX2__
    __m256 v_alpha = _mm256_set1_ps(alpha);
    for (; j + 7 < n; j += 8) {
      __m256 v_src = _mm256_loadu_ps(src + j);
      __m256 v_hid = _mm256_loadu_ps(hid + j);
      _mm256_storeu_ps(dest + j, _mm256_add_ps(v_src, _mm256_mul_ps(v_alpha, v_hid)));
    }
#endif
    for (; j < n; ++j) {
      dest[j] = src[j] + alpha * hid[j];
    }
  }
  return Predict(input);
}

std::valarray<float>& Lstm::Predict(unsigned int input) {
  for (unsigned int i = 0; i < layers_.size(); ++i) {
    auto start = begin(hidden_) + i * num_cells_;
    std::copy(start, start + num_cells_, begin(layer_input_[epoch_][i]) +
        input_size_);
    layers_[i]->ForwardPass(layer_input_[epoch_][i], input, &hidden_, i *
        num_cells_);
    if (i < layers_.size() - 1) {
      auto start2 = begin(layer_input_[epoch_][i + 1]) + num_cells_ +
          input_size_;
      std::copy(start, start + num_cells_, start2);
    }
  }
  float max_out = -1e20f;
  for (unsigned int i = 0; i < output_size_; ++i) {
    const float* hid_ptr = &hidden_[0];
    const float* out_ptr = &output_layer_[epoch_][i][0];
    float sum = dot_product_f(hid_ptr, out_ptr, hidden_.size());
    output_[epoch_][i] = sum;
    if (sum > max_out) max_out = sum;
  }
  float total_sum = 0;
  for (unsigned int i = 0; i < output_size_; ++i) {
    float val = exp(output_[epoch_][i] - max_out);
    output_[epoch_][i] = val;
    total_sum += val;
  }
  float inv_sum = 1.0f / total_sum;
  for (unsigned int i = 0; i < output_size_; ++i) {
    output_[epoch_][i] *= inv_sum;
  }
  int epoch = epoch_;
  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
  return output_[epoch];
}
