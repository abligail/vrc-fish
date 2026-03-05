#pragma once

#include <opencv2/core/types.hpp>

#include "../config/app_config.h"

namespace engine {

struct ControlInput {
	int fishY{};
	int sliderCenterY{};
	int sliderHeight{};
	double smoothSliderVelocity{};
	double smoothFishVelocity{};
	double smoothFishAccel{};
	bool holding{};
	cv::Rect fixedTrackRoi{};
	double lastDtRatio{1.0};
	double prevDeviation{};
	bool hasPrevDeviation{};
};

struct ControlDecision {
	bool wantHold{};
	bool reactiveTriggered{};
	double costPress{};
	double costRelease{};
	double deviationForNext{};
	bool hasDeviationForNext{};
};

ControlDecision computeControlDecision(const ControlInput& input, const AppConfig& config);

}  // namespace engine
