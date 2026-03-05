#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "../config/app_config.h"

namespace engine {

struct GrayTpl {
	cv::Mat gray;
	cv::Mat mask;  // Non-empty when alpha mask is available.

	bool empty() const { return gray.empty(); }
	int cols() const { return gray.cols; }
	int rows() const { return gray.rows; }
	cv::Size size() const { return gray.size(); }
};

struct TemplateStore {
	GrayTpl biteExclBottom;
	GrayTpl biteExclFull;
	GrayTpl minigameBarFull;
	GrayTpl fishIcon;
	GrayTpl fishIconAlt;
	GrayTpl fishIconAlt2;
	std::vector<GrayTpl> fishIcons;
	std::vector<std::string> fishIconFiles;
	GrayTpl playerSlider;
};

GrayTpl loadGrayTplFromFile(const std::string& path);
GrayTpl tryLoadGrayTplFromFile(const std::string& path);
std::vector<std::string> listFishAltIconFiles(const std::string& dir);
TemplateStore loadTemplateStore(const AppConfig& config);

}  // namespace engine
