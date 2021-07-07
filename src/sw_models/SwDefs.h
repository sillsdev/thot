#pragma once

#include "nlp_common/PositionIndex.h"

#include <cmath>

constexpr double SW_PROB_SMOOTH = 1e-7;
const double SW_LOG_PROB_SMOOTH = log(SW_PROB_SMOOTH);
constexpr PositionIndex IBM_SWM_MAX_SENT_LENGTH = 1024;
constexpr PositionIndex HMM_SWM_MAX_SENT_LENGTH = 200;
constexpr PositionIndex IBM4_SWM_MAX_SENT_LENGTH = 200;
