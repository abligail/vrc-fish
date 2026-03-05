#include "template_store.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../infra/fs/path_utils.h"

namespace {

std::string toLowerAscii(std::string s) {
	for (char& c : s) {
		if (c >= 'A' && c <= 'Z') {
			c = static_cast<char>(c - 'A' + 'a');
		}
	}
	return s;
}

bool isFishAltIconFilename(const std::string& file) {
	const std::string f = toLowerAscii(file);
	const std::string prefix = "fish_icon_alt";
	const std::string suffix = ".png";
	if (f.size() < prefix.size() + suffix.size()) return false;
	if (f.rfind(prefix, 0) != 0) return false;
	if (f.compare(f.size() - suffix.size(), suffix.size(), suffix) != 0) return false;
	const std::string mid = f.substr(prefix.size(), f.size() - prefix.size() - suffix.size());
	if (mid.empty()) return true;
	for (char c : mid) {
		if (c < '0' || c > '9') return false;
	}
	return true;
}

int parseFishAltIconIndex(const std::string& file) {
	if (!isFishAltIconFilename(file)) {
		return -1;
	}
	const std::string f = toLowerAscii(file);
	const std::string prefix = "fish_icon_alt";
	const std::string suffix = ".png";
	const std::string mid = f.substr(prefix.size(), f.size() - prefix.size() - suffix.size());
	if (mid.empty()) {
		return -1;
	}
	int value = 0;
	for (char c : mid) {
		const int digit = static_cast<int>(c - '0');
		if (value > 100000000) {
			return -1;
		}
		value = value * 10 + digit;
	}
	return value;
}

std::vector<std::string> listFilesByWildcard(const std::string& dir, const std::string& wildcard) {
	std::vector<std::string> out;
	const std::string query = infra::fs::joinPath(dir, wildcard);
	WIN32_FIND_DATAA ffd{};
	HANDLE hFind = FindFirstFileA(query.c_str(), &ffd);
	if (hFind == INVALID_HANDLE_VALUE) {
		return out;
	}
	do {
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
			continue;
		}
		if (ffd.cFileName[0] == '\0') {
			continue;
		}
		out.emplace_back(ffd.cFileName);
	} while (FindNextFileA(hFind, &ffd));
	FindClose(hFind);
	return out;
}

}  // namespace

namespace engine {

GrayTpl loadGrayTplFromFile(const std::string& path) {
	cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (raw.empty()) {
		std::cout << "template load failed: " << path << std::endl;
		std::exit(0);
	}

	GrayTpl tpl{};
	if (raw.channels() == 4) {
		std::vector<cv::Mat> channels;
		cv::split(raw, channels);
		cv::Mat alpha = channels[3];
		double minA = 0.0;
		double maxA = 0.0;
		cv::minMaxLoc(alpha, &minA, &maxA);

		cv::Mat bgr;
		cv::cvtColor(raw, bgr, cv::COLOR_BGRA2BGR);
		cv::cvtColor(bgr, tpl.gray, cv::COLOR_BGR2GRAY);
		if (minA < 255.0) {
			cv::threshold(alpha, tpl.mask, 0, 255, cv::THRESH_BINARY);
			std::cout << "template loaded (with mask): " << path << std::endl;
		} else {
			std::cout << "template loaded: " << path << std::endl;
		}
	} else {
		if (raw.channels() == 1) {
			tpl.gray = raw;
		} else {
			cv::cvtColor(raw, tpl.gray, cv::COLOR_BGR2GRAY);
		}
		std::cout << "template loaded: " << path << std::endl;
	}

	return tpl;
}

GrayTpl tryLoadGrayTplFromFile(const std::string& path) {
	cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
	if (raw.empty()) {
		std::cout << "template load failed (ignored): " << path << std::endl;
		return GrayTpl{};
	}

	GrayTpl tpl{};
	if (raw.channels() == 4) {
		std::vector<cv::Mat> channels;
		cv::split(raw, channels);
		cv::Mat alpha = channels[3];
		double minA = 0.0;
		double maxA = 0.0;
		cv::minMaxLoc(alpha, &minA, &maxA);

		cv::Mat bgr;
		cv::cvtColor(raw, bgr, cv::COLOR_BGRA2BGR);
		cv::cvtColor(bgr, tpl.gray, cv::COLOR_BGR2GRAY);
		if (minA < 255.0) {
			cv::threshold(alpha, tpl.mask, 0, 255, cv::THRESH_BINARY);
			std::cout << "template loaded (with mask): " << path << std::endl;
		} else {
			std::cout << "template loaded: " << path << std::endl;
		}
	} else {
		if (raw.channels() == 1) {
			tpl.gray = raw;
		} else {
			cv::cvtColor(raw, tpl.gray, cv::COLOR_BGR2GRAY);
		}
		std::cout << "template loaded: " << path << std::endl;
	}

	return tpl;
}

std::vector<std::string> listFishAltIconFiles(const std::string& dir) {
	std::vector<std::string> files = listFilesByWildcard(dir, "fish_icon_alt*.png");
	std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
		const bool aOk = isFishAltIconFilename(a);
		const bool bOk = isFishAltIconFilename(b);
		if (aOk != bOk) return aOk;
		if (!aOk) return toLowerAscii(a) < toLowerAscii(b);

		const int ai = parseFishAltIconIndex(a);
		const int bi = parseFishAltIconIndex(b);
		if (ai != bi) {
			if (ai < 0) return true;
			if (bi < 0) return false;
			return ai < bi;
		}
		return toLowerAscii(a) < toLowerAscii(b);
	});
	files.erase(std::unique(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
		return toLowerAscii(a) == toLowerAscii(b);
	}), files.end());
	return files;
}

TemplateStore loadTemplateStore(const AppConfig& config) {
	TemplateStore store{};
	store.biteExclBottom = loadGrayTplFromFile(infra::fs::joinPath(config.resource_dir, config.tpl_bite_exclamation_bottom));
	store.biteExclFull = loadGrayTplFromFile(infra::fs::joinPath(config.resource_dir, config.tpl_bite_exclamation_full));
	store.minigameBarFull = loadGrayTplFromFile(infra::fs::joinPath(config.resource_dir, config.tpl_minigame_bar_full));
	store.playerSlider = loadGrayTplFromFile(infra::fs::joinPath(config.resource_dir, config.tpl_player_slider));

	store.fishIcons.clear();
	store.fishIconFiles.clear();
	std::vector<std::string> seenFishFilesLower;

	auto addFishTplFile = [&](const std::string& file, bool required, GrayTpl* legacyOut) {
		if (file.empty()) {
			return;
		}
		const std::string key = toLowerAscii(file);
		if (std::find(seenFishFilesLower.begin(), seenFishFilesLower.end(), key) != seenFishFilesLower.end()) {
			return;
		}
		GrayTpl tpl = required
			? loadGrayTplFromFile(infra::fs::joinPath(config.resource_dir, file))
			: tryLoadGrayTplFromFile(infra::fs::joinPath(config.resource_dir, file));
		if (tpl.empty()) {
			return;
		}
		seenFishFilesLower.push_back(key);
		if (legacyOut != nullptr) {
			*legacyOut = tpl;
		}
		store.fishIcons.push_back(tpl);
		store.fishIconFiles.push_back(file);
	};

	addFishTplFile(config.tpl_fish_icon, true, &store.fishIcon);
	addFishTplFile(config.tpl_fish_icon_alt, false, &store.fishIconAlt);
	addFishTplFile(config.tpl_fish_icon_alt2, false, &store.fishIconAlt2);
	for (const std::string& file : listFishAltIconFiles(config.resource_dir)) {
		if (!isFishAltIconFilename(file)) {
			continue;
		}
		addFishTplFile(file, false, nullptr);
	}

	if (store.fishIcons.empty()) {
		std::cout << "no fish icon templates were loaded; check Resource-VRChat and tpl_fish_icon." << std::endl;
		std::exit(0);
	}

	return store;
}

}  // namespace engine
