#pragma once

#include <opencv2/core/mat.hpp>

#include "../config/app_config.h"
#include "../core/types.h"
#include "matcher.h"
#include "template_store.h"

namespace engine {

bool detectBite(
	const cv::Mat& gray,
	const TemplateStore& templates,
	const AppConfig& config,
	TplMatch* matchOut = nullptr);

bool detectFishAndSliderFast(
	const cv::Mat& gray,
	const cv::Rect& barRect,
	const TemplateStore& templates,
	const AppConfig& config,
	double trackScale,
	double trackAngleDeg,
	int cachedFishTplIdx,
	FishSliderResult* result);

bool detectFishAndSliderFull(
	const cv::Mat& gray,
	const cv::Rect& barRect,
	const TemplateStore& templates,
	const AppConfig& config,
	double trackScale,
	double trackAngleDeg,
	FishSliderResult* result,
	int* bestTplIdxOut);

}  // namespace engine
