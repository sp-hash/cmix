#include "byte-mixer.h"
#include <algorithm>
#include <cstring>

ByteMixer::ByteMixer(unsigned int num_models, const unsigned int& bit_context,
    const std::vector<bool>& vocab, unsigned int vocab_size, Lstm* lstm) :
    ByteModel(vocab), lstm_(lstm), byte_(bit_context), byte_map_(0, 256),
    inputs_(0.0, vocab_size), num_models_(num_models), vocab_size_(vocab_size),
    offset_(0) {
  for (int i = 0; i < 256; ++i) {
    byte_map_[i] = offset_;
    if (vocab_[i]) ++offset_;
  }
  // build list of vocab indices for fast updates
  vocab_indices_.reserve(vocab_size);
  for (int i = 0; i < 256; ++i) if (vocab_[i]) vocab_indices_.push_back(i);
  offset_ = 0;
}

void ByteMixer::SetInput(int index, float val) {
  if (!vocab_[index]) return;
  inputs_[offset_] += val;
  ++offset_;
  if (offset_ == vocab_size_) offset_ = 0;
}

void ByteMixer::AddInputs(const std::valarray<float>& probs) {
  const float* p = &probs[0];
  float* inp = &inputs_[0];
  for (int i = 0; i < 256; ++i) {
    if (vocab_[i]) {
      inp[offset_] += p[i];
      ++offset_;
    }
  }
  if (offset_ >= vocab_size_) offset_ = 0;
}

void ByteMixer::ByteUpdate() {
  inputs_ *= 2.0f / num_models_;
  lstm_->SetInput(inputs_);
  inputs_ = 0;
  const auto& output = lstm_->Perceive(byte_map_[byte_]);
  const float* out_ptr = &output[0];
  // zero probs and copy vocab outputs into the correct positions
  std::fill_n(&probs_[0], 256, 0.0f);
  for (unsigned int k = 0; k < vocab_indices_.size(); ++k) {
    probs_[vocab_indices_[k]] = out_ptr[k];
  }
  ByteModel::UpdateProbs(probs_);
}
