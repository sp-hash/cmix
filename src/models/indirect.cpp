#include "indirect.h"
#include <stdlib.h>

Indirect::Indirect(const State& state,
    const unsigned long long& byte_context,
    const unsigned int& bit_context, float delta,
    std::vector<unsigned char>& map) :  byte_context_(byte_context),
    bit_context_(bit_context), map_index_(0), map_offset_(0),
    divisor_(1.0 / delta), state_(state), map_(map) {
  // Guard against small maps: original code assumes map_.size() > 257.
  // If the map is too small, fall back to offset 0 to avoid modulo by zero
  // and out-of-range indexing.
  if (map_.size() > 257) {
    map_offset_ = rand() % (map_.size() - 257);
  } else {
    map_offset_ = 0;
  }
  for (int i = 0; i < 256; ++i) {
    predictions_[i] = state_.InitProbability(i);
  }
}

const std::valarray<float>& Indirect::Predict() {
  // Advance map_index_ safely using modulo to avoid out-of-range access.
  if (map_.empty()) {
    outputs_[0] = 0.0f;
    return outputs_;
  }
  map_index_ = (map_index_ + (bit_context_ % map_.size())) % map_.size();
  outputs_[0] = predictions_[map_[map_index_]];
  return outputs_;
}

void Indirect::Perceive(int bit) {
  if (map_.empty()) return;
  int state = map_[map_index_];
  predictions_[state] += (bit - predictions_[state]) * divisor_;
  map_[map_index_] = state_.Next(state, bit);
  // Rewind map_index_ safely using modulo arithmetic.
  map_index_ = (map_index_ + map_.size() - (bit_context_ % map_.size())) % map_.size();
}

void Indirect::ByteUpdate() {
  // Compute starting index for this byte. Original implementation assumed
  // map_.size() > 257 and used (map_.size() - 257) as modulus so that
  // subsequent bit-context offsets stay within bounds. If the map is small,
  // fall back to index 0 and avoid modulo by zero.
  if (map_.size() > 257) {
    map_index_ = (257 * byte_context_ + map_offset_) % (map_.size() - 257);
  } else if (!map_.empty()) {
    map_index_ = 0;
  } else {
    map_index_ = 0;
  }
}
