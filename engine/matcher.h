#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "../config/app_config.h"
#include "../core/types.h"
#include "template_store.h"

namespace engine {

cv::Rect clampRect(cv::Rect rect, const cv::Size& bounds);
cv::Rect centerThirdStripRoi(const cv::Size& bounds);

TplMatch matchBest(const cv::Mat& srcGray, const GrayTpl& tpl, int defaultMethod = cv::TM_CCOEFF_NORMED);
TplMatch matchBestRoi(const cv::Mat& srcGray, const GrayTpl& tpl, cv::Rect roi, int method = cv::TM_CCOEFF_NORMED);
TplMatch matchBestRoiAtScaleAndAngle(
	const cv::Mat& srcGray,
	const GrayTpl& tpl,
	cv::Rect roi,
	double scale,
	double angleDeg,
	int method = cv::TM_CCOEFF_NORMED);

std::vector<double> buildScaleRange(double minScale, double maxScale, double step, int maxCount = 128);
std::vector<double> buildAngleRange(double minAngle, double maxAngle, double step, int maxCount = 256);

TplMatch matchBestRoiTrackBarAutoScale(
	const cv::Mat& srcGray,
	const GrayTpl& tpl,
	cv::Rect roi,
	const AppConfig& config,
	int method = cv::TM_CCOEFF_NORMED,
	double* bestScaleOut = nullptr,
	double* bestAngleOut = nullptr);

}  // namespace engine
