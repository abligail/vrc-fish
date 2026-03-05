#include "matcher.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace {

struct ScaleMatch {
	double scale = 1.0;
	TplMatch match{};
};

struct AngleMatch {
	double angleDeg = 0.0;
	TplMatch match{};
};

void sortUniqueScales(std::vector<double>& scales) {
	std::sort(scales.begin(), scales.end());
	std::vector<double> uniq;
	uniq.reserve(scales.size());
	for (double s : scales) {
		if (!std::isfinite(s) || s <= 0.0) {
			continue;
		}
		if (uniq.empty() || std::abs(uniq.back() - s) > 1e-6) {
			uniq.push_back(s);
		}
	}
	scales.swap(uniq);
}

void sortUniqueAngles(std::vector<double>& angles) {
	std::sort(angles.begin(), angles.end());
	std::vector<double> uniq;
	uniq.reserve(angles.size());
	for (double a : angles) {
		if (!std::isfinite(a)) {
			continue;
		}
		if (uniq.empty() || std::abs(uniq.back() - a) > 1e-6) {
			uniq.push_back(a);
		}
	}
	angles.swap(uniq);
}

engine::GrayTpl makeScaledTpl(const engine::GrayTpl& tpl, double scale) {
	engine::GrayTpl scaled{};
	if (tpl.empty()) {
		return scaled;
	}

	if (!std::isfinite(scale) || scale <= 0.0 || std::abs(scale - 1.0) < 1e-6) {
		scaled.gray = tpl.gray;
		scaled.mask = tpl.mask;
		return scaled;
	}

	const int tw = std::max(1, static_cast<int>(std::round(tpl.gray.cols * scale)));
	const int th = std::max(1, static_cast<int>(std::round(tpl.gray.rows * scale)));
	cv::resize(tpl.gray, scaled.gray, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
	if (!tpl.mask.empty()) {
		cv::resize(tpl.mask, scaled.mask, cv::Size(tw, th), 0, 0, cv::INTER_NEAREST);
	}
	return scaled;
}

engine::GrayTpl rotateTplKeepAll(const engine::GrayTpl& tpl, double angleDeg) {
	engine::GrayTpl out{};
	if (tpl.empty()) {
		return out;
	}
	if (!std::isfinite(angleDeg) || std::abs(angleDeg) < 1e-6) {
		out.gray = tpl.gray;
		out.mask = tpl.mask;
		return out;
	}

	const int w = tpl.gray.cols;
	const int h = tpl.gray.rows;
	cv::Point2f center(static_cast<float>(w) / 2.0f, static_cast<float>(h) / 2.0f);

	cv::RotatedRect rr(center, cv::Size2f(static_cast<float>(w), static_cast<float>(h)), static_cast<float>(angleDeg));
	cv::Rect2f bbox = rr.boundingRect2f();
	const int outW = std::max(1, static_cast<int>(std::ceil(bbox.width)));
	const int outH = std::max(1, static_cast<int>(std::ceil(bbox.height)));

	cv::Mat M = cv::getRotationMatrix2D(center, angleDeg, 1.0);
	M.at<double>(0, 2) += static_cast<double>(outW) / 2.0 - center.x;
	M.at<double>(1, 2) += static_cast<double>(outH) / 2.0 - center.y;

	cv::warpAffine(tpl.gray, out.gray, M, cv::Size(outW, outH), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
	if (!tpl.mask.empty()) {
		cv::warpAffine(tpl.mask, out.mask, M, cv::Size(outW, outH), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
	}
	return out;
}

TplMatch matchBestRoiAtScale(
	const cv::Mat& srcGray,
	const engine::GrayTpl& tpl,
	cv::Rect roi,
	double scale,
	int method = cv::TM_CCOEFF_NORMED) {
	TplMatch out{};
	if (srcGray.empty() || tpl.empty()) {
		return out;
	}

	roi = engine::clampRect(roi, srcGray.size());
	if (roi.width <= 0 || roi.height <= 0) {
		return out;
	}

	cv::Mat sub = srcGray(roi);
	engine::GrayTpl scaled = makeScaledTpl(tpl, scale);
	if (sub.cols < scaled.gray.cols || sub.rows < scaled.gray.rows) {
		return out;
	}

	out = engine::matchBest(sub, scaled, method);
	out.topLeft += roi.tl();
	out.center += roi.tl();
	out.rect.x += roi.x;
	out.rect.y += roi.y;
	return out;
}

TplMatch matchBestRoiMultiScaleCoarseToFine(
	const cv::Mat& srcGray,
	const engine::GrayTpl& tpl,
	cv::Rect roi,
	const std::vector<double>& coarseScales,
	int refineTopK,
	double refineRadius,
	double refineStep,
	int method,
	double* bestScaleOut) {
	TplMatch best{};
	if (srcGray.empty() || tpl.empty()) {
		return best;
	}

	double bestScale = 1.0;
	std::vector<ScaleMatch> coarseMatches;
	coarseMatches.reserve(coarseScales.size());

	double coarseMin = 0.0;
	double coarseMax = 0.0;
	bool hasCoarseMinMax = false;

	for (double s : coarseScales) {
		if (!std::isfinite(s) || s <= 0.0) {
			continue;
		}

		if (!hasCoarseMinMax) {
			coarseMin = coarseMax = s;
			hasCoarseMinMax = true;
		} else {
			if (s < coarseMin) coarseMin = s;
			if (s > coarseMax) coarseMax = s;
		}

		TplMatch m = matchBestRoiAtScale(srcGray, tpl, roi, s, method);
		coarseMatches.push_back(ScaleMatch{ s, m });
		if (m.score > best.score) {
			best = m;
			bestScale = s;
		}
	}

	if (coarseMatches.empty()) {
		best = matchBestRoiAtScale(srcGray, tpl, roi, 1.0, method);
		bestScale = 1.0;
		if (bestScaleOut != nullptr) {
			*bestScaleOut = bestScale;
		}
		return best;
	}

	if (refineTopK < 1) refineTopK = 0;
	if (!std::isfinite(refineRadius) || refineRadius <= 0.0) refineTopK = 0;
	if (!std::isfinite(refineStep) || refineStep <= 0.0) refineTopK = 0;
	if (!hasCoarseMinMax || coarseMax <= 0.0 || coarseMin <= 0.0) refineTopK = 0;

	if (refineTopK > 0) {
		std::sort(coarseMatches.begin(), coarseMatches.end(), [](const ScaleMatch& a, const ScaleMatch& b) {
			return a.match.score > b.match.score;
		});

		std::vector<double> refineScales;
		refineScales.reserve(static_cast<size_t>(refineTopK) * 16);

		for (int i = 0; i < static_cast<int>(coarseMatches.size()) && i < refineTopK; i++) {
			const ScaleMatch& cand = coarseMatches[static_cast<size_t>(i)];
			if (cand.match.score <= 0.0) {
				break;
			}

			double s1 = cand.scale - refineRadius;
			double s2 = cand.scale + refineRadius;
			if (s1 < coarseMin) s1 = coarseMin;
			if (s2 > coarseMax) s2 = coarseMax;
			if (s2 <= s1) {
				continue;
			}

			std::vector<double> seg = engine::buildScaleRange(s1, s2, refineStep, 96);
			refineScales.insert(refineScales.end(), seg.begin(), seg.end());
		}

		sortUniqueScales(refineScales);
		for (double s : refineScales) {
			TplMatch m = matchBestRoiAtScale(srcGray, tpl, roi, s, method);
			if (m.score > best.score) {
				best = m;
				bestScale = s;
			}
		}
	}

	if (bestScaleOut != nullptr) {
		*bestScaleOut = bestScale;
	}
	return best;
}

TplMatch matchBestRoiMultiAngleAtScaleCoarseToFine(
	const cv::Mat& srcGray,
	const engine::GrayTpl& tpl,
	cv::Rect roi,
	double scale,
	const std::vector<double>& coarseAngles,
	int refineTopK,
	double refineRadius,
	double refineStep,
	int method,
	double* bestAngleOut) {
	TplMatch best{};
	if (srcGray.empty() || tpl.empty()) {
		return best;
	}

	engine::GrayTpl scaledTpl = makeScaledTpl(tpl, scale);
	if (scaledTpl.empty()) {
		return best;
	}

	double bestAngle = 0.0;
	std::vector<AngleMatch> coarseMatches;
	coarseMatches.reserve(coarseAngles.size());

	double coarseMin = 0.0;
	double coarseMax = 0.0;
	bool hasCoarseMinMax = false;

	for (double a : coarseAngles) {
		if (!std::isfinite(a)) {
			continue;
		}
		if (!hasCoarseMinMax) {
			coarseMin = coarseMax = a;
			hasCoarseMinMax = true;
		} else {
			if (a < coarseMin) coarseMin = a;
			if (a > coarseMax) coarseMax = a;
		}

		engine::GrayTpl rotated = rotateTplKeepAll(scaledTpl, a);
		TplMatch m = engine::matchBestRoi(srcGray, rotated, roi, method);
		coarseMatches.push_back(AngleMatch{ a, m });
		if (m.score > best.score) {
			best = m;
			bestAngle = a;
		}
	}

	if (coarseMatches.empty()) {
		engine::GrayTpl rotated = rotateTplKeepAll(scaledTpl, 0.0);
		best = engine::matchBestRoi(srcGray, rotated, roi, method);
		bestAngle = 0.0;
		if (bestAngleOut != nullptr) {
			*bestAngleOut = bestAngle;
		}
		return best;
	}

	if (refineTopK < 1) refineTopK = 0;
	if (!std::isfinite(refineRadius) || refineRadius <= 0.0) refineTopK = 0;
	if (!std::isfinite(refineStep) || refineStep <= 0.0) refineTopK = 0;
	if (!hasCoarseMinMax) refineTopK = 0;

	if (refineTopK > 0) {
		std::sort(coarseMatches.begin(), coarseMatches.end(), [](const AngleMatch& a, const AngleMatch& b) {
			return a.match.score > b.match.score;
		});

		std::vector<double> refineAngles;
		refineAngles.reserve(static_cast<size_t>(refineTopK) * 16);

		for (int i = 0; i < static_cast<int>(coarseMatches.size()) && i < refineTopK; i++) {
			const AngleMatch& cand = coarseMatches[static_cast<size_t>(i)];
			if (cand.match.score <= 0.0) {
				break;
			}

			double a1 = cand.angleDeg - refineRadius;
			double a2 = cand.angleDeg + refineRadius;
			if (a1 < coarseMin) a1 = coarseMin;
			if (a2 > coarseMax) a2 = coarseMax;
			if (a2 <= a1) {
				continue;
			}

			std::vector<double> seg = engine::buildAngleRange(a1, a2, refineStep, 128);
			refineAngles.insert(refineAngles.end(), seg.begin(), seg.end());
		}

		sortUniqueAngles(refineAngles);
		for (double a : refineAngles) {
			engine::GrayTpl rotated = rotateTplKeepAll(scaledTpl, a);
			TplMatch m = engine::matchBestRoi(srcGray, rotated, roi, method);
			if (m.score > best.score) {
				best = m;
				bestAngle = a;
			}
		}
	}

	if (bestAngleOut != nullptr) {
		*bestAngleOut = bestAngle;
	}
	return best;
}

std::vector<double> buildTrackBarScales(const AppConfig& config) {
	std::vector<double> scales;
	if (std::isfinite(config.track_scale_min) && std::isfinite(config.track_scale_max)
		&& std::isfinite(config.track_scale_step)
		&& config.track_scale_min > 0.0 && config.track_scale_max > 0.0 && config.track_scale_step > 0.0
		&& config.track_scale_max >= config.track_scale_min) {
		scales = engine::buildScaleRange(config.track_scale_min, config.track_scale_max, config.track_scale_step, 128);
	}
	if (scales.empty()) {
		scales = {
			config.track_scale_1,
			config.track_scale_2,
			config.track_scale_3,
			config.track_scale_4,
		};
		sortUniqueScales(scales);
	}
	return scales;
}

}  // namespace

namespace engine {

cv::Rect clampRect(cv::Rect rect, const cv::Size& bounds) {
	if (rect.x < 0) {
		rect.width += rect.x;
		rect.x = 0;
	}
	if (rect.y < 0) {
		rect.height += rect.y;
		rect.y = 0;
	}
	if (rect.x + rect.width > bounds.width) {
		rect.width = bounds.width - rect.x;
	}
	if (rect.y + rect.height > bounds.height) {
		rect.height = bounds.height - rect.y;
	}
	if (rect.width < 0) rect.width = 0;
	if (rect.height < 0) rect.height = 0;
	return rect;
}

cv::Rect centerThirdStripRoi(const cv::Size& bounds) {
	const int w = bounds.width;
	const int h = bounds.height;
	const int x1 = w / 3;
	const int x2 = (w * 2) / 3;
	return clampRect(cv::Rect(x1, 0, x2 - x1, h), bounds);
}

TplMatch matchBest(const cv::Mat& srcGray, const GrayTpl& tpl, int defaultMethod) {
	TplMatch out{};
	const cv::Mat& tplGray = tpl.gray;
	if (srcGray.empty() || tplGray.empty()) {
		return out;
	}
	if (srcGray.cols < tplGray.cols || srcGray.rows < tplGray.rows) {
		return out;
	}

	cv::Mat result;
	const int method = defaultMethod;
	if (!tpl.mask.empty()) {
		cv::matchTemplate(srcGray, tplGray, result, method, tpl.mask);
	} else {
		cv::matchTemplate(srcGray, tplGray, result, method);
	}

	double minVal = 0.0;
	double maxVal = 0.0;
	cv::Point minLoc{};
	cv::Point maxLoc{};
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	cv::Point bestLoc = maxLoc;
	double score = maxVal;
	if (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED) {
		bestLoc = minLoc;
		score = 1.0 - minVal;
	}
	if (!std::isfinite(score)) {
		score = 0.0;
	}

	out.topLeft = bestLoc;
	out.rect = cv::Rect(bestLoc.x, bestLoc.y, tplGray.cols, tplGray.rows);
	out.center = cv::Point(bestLoc.x + tplGray.cols / 2, bestLoc.y + tplGray.rows / 2);
	out.score = score;
	return out;
}

TplMatch matchBestRoi(const cv::Mat& srcGray, const GrayTpl& tpl, cv::Rect roi, int method) {
	TplMatch out{};
	if (srcGray.empty() || tpl.empty()) {
		return out;
	}

	roi = clampRect(roi, srcGray.size());
	if (roi.width < tpl.cols() || roi.height < tpl.rows()) {
		return out;
	}

	cv::Mat sub = srcGray(roi);
	out = matchBest(sub, tpl, method);
	out.topLeft += roi.tl();
	out.center += roi.tl();
	out.rect.x += roi.x;
	out.rect.y += roi.y;
	return out;
}

TplMatch matchBestRoiAtScaleAndAngle(
	const cv::Mat& srcGray,
	const GrayTpl& tpl,
	cv::Rect roi,
	double scale,
	double angleDeg,
	int method) {
	if (!std::isfinite(scale) || scale <= 0.0) {
		scale = 1.0;
	}
	if (!std::isfinite(angleDeg)) {
		angleDeg = 0.0;
	}
	GrayTpl scaled = makeScaledTpl(tpl, scale);
	GrayTpl rotated = rotateTplKeepAll(scaled, angleDeg);
	return matchBestRoi(srcGray, rotated, roi, method);
}

std::vector<double> buildScaleRange(double minScale, double maxScale, double step, int maxCount) {
	std::vector<double> scales;
	if (!std::isfinite(minScale) || !std::isfinite(maxScale) || !std::isfinite(step)) {
		return scales;
	}
	if (step <= 0.0) {
		return scales;
	}
	if (minScale > maxScale) {
		std::swap(minScale, maxScale);
	}
	if (minScale <= 0.0 || maxScale <= 0.0) {
		return scales;
	}
	if (maxCount < 2) {
		maxCount = 2;
	}

	const double span = maxScale - minScale;
	int count = static_cast<int>(std::floor(span / step + 1e-9)) + 1;
	if (count < 1) {
		count = 1;
	}
	if (count > maxCount) {
		step = (span <= 0.0) ? step : (span / static_cast<double>(maxCount - 1));
		count = maxCount;
	}

	scales.reserve(count + 1);
	for (int i = 0; i < count; i++) {
		double s = minScale + step * static_cast<double>(i);
		if (!std::isfinite(s) || s <= 0.0) {
			continue;
		}
		scales.push_back(s);
	}
	if (!scales.empty()) {
		const double last = scales.back();
		if (std::abs(last - maxScale) > step * 0.25) {
			scales.push_back(maxScale);
		}
	}
	sortUniqueScales(scales);
	return scales;
}

std::vector<double> buildAngleRange(double minAngle, double maxAngle, double step, int maxCount) {
	std::vector<double> angles;
	if (!std::isfinite(minAngle) || !std::isfinite(maxAngle) || !std::isfinite(step)) {
		return angles;
	}
	if (step <= 0.0) {
		return angles;
	}
	if (minAngle > maxAngle) {
		std::swap(minAngle, maxAngle);
	}
	if (maxCount < 2) {
		maxCount = 2;
	}

	const double span = maxAngle - minAngle;
	int count = static_cast<int>(std::floor(span / step + 1e-9)) + 1;
	if (count < 1) {
		count = 1;
	}
	if (count > maxCount) {
		step = (span <= 0.0) ? step : (span / static_cast<double>(maxCount - 1));
		count = maxCount;
	}

	angles.reserve(count + 1);
	for (int i = 0; i < count; i++) {
		double a = minAngle + step * static_cast<double>(i);
		if (!std::isfinite(a)) {
			continue;
		}
		angles.push_back(a);
	}
	if (!angles.empty()) {
		const double last = angles.back();
		if (std::abs(last - maxAngle) > step * 0.25) {
			angles.push_back(maxAngle);
		}
	}
	sortUniqueAngles(angles);
	return angles;
}

TplMatch matchBestRoiTrackBarAutoScale(
	const cv::Mat& srcGray,
	const GrayTpl& tpl,
	cv::Rect roi,
	const AppConfig& config,
	int method,
	double* bestScaleOut,
	double* bestAngleOut) {
	const std::vector<double> coarse = buildTrackBarScales(config);
	double bestScale = 1.0;
	TplMatch best = matchBestRoiMultiScaleCoarseToFine(
		srcGray,
		tpl,
		roi,
		coarse,
		config.track_scale_refine_topk,
		config.track_scale_refine_radius,
		config.track_scale_refine_step,
		method,
		&bestScale);

	double bestAngle = 0.0;
	if (std::isfinite(config.track_angle_min) && std::isfinite(config.track_angle_max)
		&& std::isfinite(config.track_angle_step)
		&& config.track_angle_step > 0.0) {
		double aMin = config.track_angle_min;
		double aMax = config.track_angle_max;
		if (aMin > aMax) {
			std::swap(aMin, aMax);
		}
		std::vector<double> angles = buildAngleRange(aMin, aMax, config.track_angle_step, 256);
		if (!angles.empty()) {
			best = matchBestRoiMultiAngleAtScaleCoarseToFine(
				srcGray,
				tpl,
				roi,
				bestScale,
				angles,
				config.track_angle_refine_topk,
				config.track_angle_refine_radius,
				config.track_angle_refine_step,
				method,
				&bestAngle);
		}
	}

	if (bestScaleOut != nullptr) {
		*bestScaleOut = bestScale;
	}
	if (bestAngleOut != nullptr) {
		*bestAngleOut = bestAngle;
	}
	return best;
}

}  // namespace engine
