#ifndef BYTE_MODEL_H
#define BYTE_MODEL_H

#include "model.h"

#include <valarray>
#include <vector>

class ByteModel : public Model {
 public:
  virtual ~ByteModel() {}
  ByteModel(const std::vector<bool>& vocab);
  const std::valarray<float>& BytePredict();
  const std::valarray<float>& Predict();
  void Perceive(int bit);
  virtual void ByteUpdate();
  void UpdateProbs(const unsigned int* frequencies);
  void UpdateProbs(const std::valarray<float>& normalized_probs);
  int ex;
 protected:
  void RefreshSums();
  int top_, mid_, bot_;
  const std::vector<bool>& vocab_;
  std::valarray<float> probs_;
  float cumulative_probs_[257];
  bool dirty_;
};

#endif

