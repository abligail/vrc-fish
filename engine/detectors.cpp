#include "detectors.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {

bool detectSliderBounds(
	const cv::Mat& gray,
	int barX,
	const cv::Rect& searchRoi,
	const AppConfig& config,
	int* sliderTopOut,
	int* sliderBottomOut,
	int* sliderCenterYOut,
	int brightnessThresh = 180,
	int minSliderHeight = 15) {
	int halfW = config.slider_detect_half_width;
	if (halfW < 0) halfW = 0;
	int x1 = std::max(searchRoi.x, barX - halfW);
	int x2 = std::min(searchRoi.x + searchRoi.width, barX + halfW + 1);
	if (x2 <= x1) return false;

	int y1 = searchRoi.y;
	int y2 = searchRoi.y + searchRoi.height;
	if (y1 < 0) y1 = 0;
	if (y2 > gray.rows) y2 = gray.rows;
	if (y2 <= y1) return false;

	struct BrightRun { int start; int len; };
	std::vector<BrightRun> runs;
	int curRunStart = -1;
	int curRunLen = 0;

	for (int y = y1; y < y2; y++) {
		const uchar* row = gray.ptr<uchar>(y);
		int sum = 0;
		for (int x = x1; x < x2; x++) {
			sum += row[x];
		}
		int avg = sum / (x2 - x1);

		if (avg >= brightnessThresh) {
			if (curRunStart < 0) curRunStart = y;
			curRunLen = y - curRunStart + 1;
		} else {
			if (curRunLen > 0) {
				runs.push_back({ curRunStart, curRunLen });
			}
			curRunStart = -1;
			curRunLen = 0;
		}
	}

	if (curRunLen > 0) {
		runs.push_back({ curRunStart, curRunLen });
	}
	if (runs.empty()) return false;

	int maxGap = config.slider_detect_merge_gap;
	if (maxGap < 0) maxGap = 0;
	std::vector<BrightRun> merged;
	merged.push_back(runs[0]);
	for (size_t i = 1; i < runs.size(); i++) {
		BrightRun& last = merged.back();
		int lastEnd = last.start + last.len;
		int gap = runs[i].start - lastEnd;
		if (gap <= maxGap) {
			last.len = (runs[i].start + runs[i].len) - last.start;
		} else {
			merged.push_back(runs[i]);
		}
	}

	int bestIdx = 0;
	for (size_t i = 1; i < merged.size(); i++) {
		if (merged[i].len > merged[bestIdx].len) {
			bestIdx = static_cast<int>(i);
		}
	}
	const int bestRunStart = merged[bestIdx].start;
	const int bestRunLen = merged[bestIdx].len;

	if (bestRunLen < minSliderHeight) return false;
	if (bestRunLen >= static_cast<int>((y2 - y1) * 0.95)) return false;

	if (sliderTopOut) *sliderTopOut = bestRunStart;
	if (sliderBottomOut) *sliderBottomOut = bestRunStart + bestRunLen;
	if (sliderCenterYOut) *sliderCenterYOut = bestRunStart + bestRunLen / 2;
	return true;
}

bool detectSliderBoundsWide(
	const cv::Mat& gray,
	const cv::Rect& searchRoi,
	const AppConfig& config,
	int* sliderTopOut,
	int* sliderBottomOut,
	int* sliderCenterYOut,
	int brightnessThresh = 180,
	int minSliderHeight = 15) {
	cv::Rect roi = engine::clampRect(searchRoi, gray.size());
	if (roi.width <= 0 || roi.height <= 0) return false;

	int x1 = roi.x;
	int x2 = roi.x + roi.width;
	int y1 = roi.y;
	int y2 = roi.y + roi.height;
	if (x2 <= x1 || y2 <= y1) return false;

	int minRowBright = roi.width / 8;
	if (minRowBright < 4) minRowBright = 4;
	if (minRowBright > roi.width) minRowBright = roi.width;

	struct BrightRun { int start; int len; };
	std::vector<BrightRun> runs;
	int curRunStart = -1;
	int curRunLen = 0;

	for (int y = y1; y < y2; y++) {
		const uchar* row = gray.ptr<uchar>(y);
		int brightCnt = 0;
		for (int x = x1; x < x2; x++) {
			if (row[x] >= brightnessThresh) {
				brightCnt++;
			}
		}

		if (brightCnt >= minRowBright) {
			if (curRunStart < 0) curRunStart = y;
			curRunLen = y - curRunStart + 1;
		} else {
			if (curRunLen > 0) {
				runs.push_back({ curRunStart, curRunLen });
			}
			curRunStart = -1;
			curRunLen = 0;
		}
	}

	if (curRunLen > 0) {
		runs.push_back({ curRunStart, curRunLen });
	}
	if (runs.empty()) return false;

	int maxGap = config.slider_detect_merge_gap;
	if (maxGap < 0) maxGap = 0;
	std::vector<BrightRun> merged;
	merged.push_back(runs[0]);
	for (size_t i = 1; i < runs.size(); i++) {
		BrightRun& last = merged.back();
		int lastEnd = last.start + last.len;
		int gap = runs[i].start - lastEnd;
		if (gap <= maxGap) {
			last.len = (runs[i].start + runs[i].len) - last.start;
		} else {
			merged.push_back(runs[i]);
		}
	}

	int bestIdx = 0;
	for (size_t i = 1; i < merged.size(); i++) {
		if (merged[i].len > merged[bestIdx].len) {
			bestIdx = static_cast<int>(i);
		}
	}
	const int bestRunStart = merged[bestIdx].start;
	const int bestRunLen = merged[bestIdx].len;

	if (bestRunLen < minSliderHeight) return false;
	if (bestRunLen >= static_cast<int>(roi.height * 0.95)) return false;

	if (sliderTopOut) *sliderTopOut = bestRunStart;
	if (sliderBottomOut) *sliderBottomOut = bestRunStart + bestRunLen;
	if (sliderCenterYOut) *sliderCenterYOut = bestRunStart + bestRunLen / 2;
	return true;
}

cv::Point2f affineTransformPoint(const cv::Mat& M, const cv::Point2f& p) {
	cv::Point2f out{};
	if (M.empty() || M.rows != 2 || M.cols != 3) {
		return p;
	}
	out.x = static_cast<float>(M.at<double>(0, 0) * p.x + M.at<double>(0, 1) * p.y + M.at<double>(0, 2));
	out.y = static_cast<float>(M.at<double>(1, 0) * p.x + M.at<double>(1, 1) * p.y + M.at<double>(1, 2));
	return out;
}

bool fillFishSliderResult(
	const cv::Mat& gray,
	const cv::Rect& roi,
	const TplMatch& fish,
	const TplMatch& slider,
	double trackAngleDeg,
	int fishTplHeightHint,
	const engine::TemplateStore& templates,
	const AppConfig& config,
	FishSliderResult* result) {
	if (result == nullptr) {
		return false;
	}
	*result = FishSliderResult{};

	result->fishScore = fish.score;
	result->sliderScore = slider.score;
	if (fish.score < config.fish_icon_threshold) {
		return false;
	}

	result->fishX = fish.center.x;
	result->fishY = fish.center.y;
	result->sliderCenterX = fish.center.x;
	result->sliderCenterY = slider.center.y;

	int effectiveMinH = config.slider_min_height;
	int fishTplH = fishTplHeightHint;
	if (fishTplH <= 0) {
		fishTplH = templates.fishIcon.rows();
	}
	if (fishTplH > 0 && effectiveMinH < fishTplH + 5) {
		effectiveMinH = fishTplH + 5;
	}

	int sliderTop = 0;
	int sliderBottom = 0;
	int sliderCenterFromColor = 0;
	{
		cv::Rect r = engine::clampRect(roi, gray.size());
		cv::Mat roiGray = gray(r);
		cv::Rect localRoi(0, 0, roiGray.cols, roiGray.rows);

		double a = trackAngleDeg;
		if (!std::isfinite(a)) {
			a = 0.0;
		}
		cv::Mat scanGray = roiGray;
		cv::Mat rotated;
		cv::Mat M;
		cv::Mat Minv;
		int barXLocal = fish.center.x - r.x;
		int barYLocal = fish.center.y - r.y;
		if (barXLocal < 0) barXLocal = 0;
		if (barXLocal >= roiGray.cols) barXLocal = roiGray.cols - 1;
		if (barYLocal < 0) barYLocal = 0;
		if (barYLocal >= roiGray.rows) barYLocal = roiGray.rows - 1;

		float barXForMap = static_cast<float>(barXLocal);
		if (std::abs(a) > 1e-6 && roiGray.cols > 1 && roiGray.rows > 1) {
			cv::Point2f c(static_cast<float>(roiGray.cols) / 2.0f, static_cast<float>(roiGray.rows) / 2.0f);
			M = cv::getRotationMatrix2D(c, -a, 1.0);
			Minv = cv::getRotationMatrix2D(c, +a, 1.0);
			cv::warpAffine(roiGray, rotated, M, roiGray.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
			scanGray = rotated;

			const cv::Point2f barRot = affineTransformPoint(M, cv::Point2f(static_cast<float>(barXLocal), static_cast<float>(barYLocal)));
			barXForMap = barRot.x;
			if (barXForMap < 0.0f) barXForMap = 0.0f;
			if (barXForMap > static_cast<float>(roiGray.cols - 1)) barXForMap = static_cast<float>(roiGray.cols - 1);
			barXLocal = static_cast<int>(std::round(barXForMap));
		}

		int t = 0;
		int b = 0;
		int cy = 0;
		bool okColor = detectSliderBoundsWide(scanGray, localRoi, config,
			&t, &b, &cy,
			config.slider_bright_thresh, effectiveMinH)
			|| detectSliderBounds(scanGray, barXLocal, localRoi, config,
				&t, &b, &cy,
				config.slider_bright_thresh, effectiveMinH);

		if (okColor) {
			if (!Minv.empty()) {
				const cv::Point2f pTop = affineTransformPoint(Minv, cv::Point2f(barXForMap, static_cast<float>(t)));
				const cv::Point2f pBot = affineTransformPoint(Minv, cv::Point2f(barXForMap, static_cast<float>(b)));
				const cv::Point2f pCy = affineTransformPoint(Minv, cv::Point2f(barXForMap, static_cast<float>(cy)));
				sliderTop = static_cast<int>(std::round(pTop.y)) + r.y;
				sliderBottom = static_cast<int>(std::round(pBot.y)) + r.y;
				sliderCenterFromColor = static_cast<int>(std::round(pCy.y)) + r.y;
			} else {
				sliderTop = t + r.y;
				sliderBottom = b + r.y;
				sliderCenterFromColor = cy + r.y;
			}

			int maxY = gray.rows > 0 ? (gray.rows - 1) : 0;
			if (sliderTop < 0) sliderTop = 0;
			if (sliderTop > maxY) sliderTop = maxY;
			if (sliderBottom < sliderTop) sliderBottom = sliderTop;
			if (sliderBottom > gray.rows) sliderBottom = gray.rows;
			if (sliderCenterFromColor < 0) sliderCenterFromColor = 0;
			if (sliderCenterFromColor > maxY) sliderCenterFromColor = maxY;

			result->sliderTop = sliderTop;
			result->sliderBottom = sliderBottom;
			result->sliderHeight = sliderBottom - sliderTop;
			result->sliderCenterY = sliderCenterFromColor;
			result->hasBounds = true;
		}
	}

	if (!result->hasBounds) {
		if (slider.score < config.slider_threshold) {
			return false;
		}
		const int tplH = templates.playerSlider.rows();
		result->sliderTop = slider.center.y - tplH / 2;
		result->sliderBottom = slider.center.y + tplH / 2;
		result->sliderHeight = tplH;
		result->hasBounds = false;
	}

	return true;
}

}  // namespace

namespace engine {

bool detectBite(const cv::Mat& gray, const TemplateStore& templates, const AppConfig& config, TplMatch* matchOut) {
	TplMatch m = matchBest(gray, templates.biteExclBottom);
	if (m.score < config.bite_threshold) {
		TplMatch m2 = matchBest(gray, templates.biteExclFull);
		if (m2.score > m.score) {
			m = m2;
		}
	}
	if (matchOut != nullptr) {
		*matchOut = m;
	}
	return m.score >= config.bite_threshold;
}

bool detectFishAndSliderFast(
	const cv::Mat& gray,
	const cv::Rect& barRect,
	const TemplateStore& templates,
	const AppConfig& config,
	double trackScale,
	double trackAngleDeg,
	int cachedFishTplIdx,
	FishSliderResult* result) {
	cv::Rect roi = clampRect(barRect, gray.size());
	if (roi.width <= 0 || roi.height <= 0) {
		return false;
	}

	double s = trackScale;
	if (!std::isfinite(s) || s <= 0.0) {
		s = 1.0;
	}

	const GrayTpl* fishTplPtr = &templates.fishIcon;
	if (!templates.fishIcons.empty()) {
		int idx = cachedFishTplIdx;
		if (idx < 0) idx = 0;
		if (idx >= static_cast<int>(templates.fishIcons.size())) idx = 0;
		fishTplPtr = &templates.fishIcons[static_cast<size_t>(idx)];
	}

	const GrayTpl& fishTpl = *fishTplPtr;
	TplMatch fish = matchBestRoiAtScaleAndAngle(gray, fishTpl, roi, s, trackAngleDeg, cv::TM_CCOEFF_NORMED);
	TplMatch slider = matchBestRoi(gray, templates.playerSlider, roi, cv::TM_CCORR_NORMED);
	const int fishTplHScaled = static_cast<int>(std::round(static_cast<double>(fishTpl.rows()) * s));
	return fillFishSliderResult(gray, roi, fish, slider, trackAngleDeg, fishTplHScaled, templates, config, result);
}

bool detectFishAndSliderFull(
	const cv::Mat& gray,
	const cv::Rect& barRect,
	const TemplateStore& templates,
	const AppConfig& config,
	double trackScale,
	double trackAngleDeg,
	FishSliderResult* result,
	int* bestTplIdxOut) {
	cv::Rect roi = clampRect(barRect, gray.size());
	if (roi.width <= 0 || roi.height <= 0) {
		return false;
	}

	double s = trackScale;
	if (!std::isfinite(s) || s <= 0.0) {
		s = 1.0;
	}

	if (templates.fishIcons.empty()) {
		return false;
	}

	TplMatch fish{};
	int bestIdx = 0;
	for (size_t i = 0; i < templates.fishIcons.size(); i++) {
		const GrayTpl& tpl = templates.fishIcons[i];
		TplMatch m = matchBestRoiAtScaleAndAngle(gray, tpl, roi, s, trackAngleDeg, cv::TM_CCOEFF_NORMED);
		if (m.score > fish.score) {
			fish = m;
			bestIdx = static_cast<int>(i);
		}
	}

	TplMatch slider = matchBestRoi(gray, templates.playerSlider, roi, cv::TM_CCORR_NORMED);
	const int fishTplHScaled = static_cast<int>(std::round(static_cast<double>(templates.fishIcons[static_cast<size_t>(bestIdx)].rows()) * s));
	const bool ok = fillFishSliderResult(gray, roi, fish, slider, trackAngleDeg, fishTplHScaled, templates, config, result);
	if (!ok) {
		return false;
	}

	if (bestTplIdxOut != nullptr) {
		*bestTplIdxOut = bestIdx;
	}
	return true;
}

}  // namespace engine
