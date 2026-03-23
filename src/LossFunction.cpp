#include "LossFunction.h"

template class LossFunction<double>;
template class MeanSquaredError<double>;
template class CrossEntropyLoss<double>;

template class LossFunction<float>;
template class MeanSquaredError<float>;
template class CrossEntropyLoss<float>;