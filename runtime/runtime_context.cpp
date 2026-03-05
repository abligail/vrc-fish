#include "runtime_context.h"

#include <cstdlib>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../gs-opencv.h"
#include "../gs-mfc.h"
#include "../infra/win/input_api.h"
#include "../infra/win/window_api.h"

namespace {

bool sameRect(const RECT& a, const RECT& b) {
	return a.left == b.left && a.top == b.top && a.right == b.right && a.bottom == b.bottom;
}

}  // namespace

namespace runtime {

RuntimeContext::RuntimeContext() = default;

RuntimeContext::~RuntimeContext() {
	shutdown();
}

bool RuntimeContext::initialize(const std::string& configPath) {
	config_ = loadAppConfig(configPath);
	paused_.store(false);

	SetConsoleTitle(L"VRChat FISH! Auto Fish (Draft)");
	std::cout << "    VRChat FISH! Auto Fish (Draft)" << std::endl << std::endl;
	std::cout << "  Note: for educational use only. Follow related rules." << std::endl << std::endl;
	std::cout << "  Hint: this mode uses left mouse hold/release control. Recommended window size: "
		<< config_.target_width << "*" << config_.target_height << std::endl << std::endl;

	hwnd_ = infra::win::findWindowByClassAndTitleContains(config_.window_class, config_.window_title_contains);
	if (!hwnd_) {
		std::cout << "VRChat window not found: class=" << config_.window_class
			<< ", title_contains=" << config_.window_title_contains << std::endl;
		return false;
	}

	HDC hdc = GetDC(hwnd_);
	if (hdc != nullptr) {
		const int dpi = GetDpiForWindow(hwnd_);
		ScaleViewportExtEx(hdc, dpi, dpi, dpi / 96, dpi / 96, nullptr);
		ScaleWindowExtEx(hdc, dpi, dpi, dpi / 96, dpi / 96, nullptr);
		ReleaseDC(hwnd_, hdc);
	}

	refreshWindowRect(true);

	stopRectThread_.store(false);
	rectThread_ = std::thread(&RuntimeContext::refreshWindowRectLoop, this);

	moveConsoleNearGameWindow();
	templates_ = engine::loadTemplateStore(config_);
	return true;
}

void RuntimeContext::shutdown() {
	stopRectThread_.store(true);
	if (rectThread_.joinable()) {
		rectThread_.join();
	}
}

AppConfig& RuntimeContext::config() {
	return config_;
}

const AppConfig& RuntimeContext::config() const {
	return config_;
}

const engine::TemplateStore& RuntimeContext::templates() const {
	return templates_;
}

HWND RuntimeContext::hwnd() const {
	return hwnd_;
}

RECT RuntimeContext::windowRect() const {
	std::lock_guard<std::mutex> lock(rectMutex_);
	return rect_;
}

cv::Mat RuntimeContext::captureBgr() const {
	return getSrc(windowRect());
}

void RuntimeContext::mouseLeftDown() const {
	infra::win::mouseLeftDown(hwnd_, windowRect(), config_.vr_debug);
}

void RuntimeContext::mouseLeftUp() const {
	infra::win::mouseLeftUp(hwnd_, windowRect(), config_.vr_debug);
}

void RuntimeContext::mouseLeftClickCentered(int delayMs) const {
	infra::win::mouseLeftClickCentered(hwnd_, windowRect(), config_.vr_debug, delayMs);
}

void RuntimeContext::mouseMoveRelative(int dx, int dy, const char* phaseTag) const {
	infra::win::mouseMoveRelative(hwnd_, windowRect(), dx, dy, config_.vr_debug, phaseTag);
}

void RuntimeContext::keyTapVk(WORD vk, int delayMs) const {
	infra::win::keyTapVk(hwnd_, windowRect(), vk, config_.vr_debug, delayMs);
}

bool RuntimeContext::isPaused() const {
	return paused_.load();
}

bool RuntimeContext::togglePause() {
	const bool next = !paused_.load();
	paused_.store(next);
	return next;
}

void RuntimeContext::setPaused(bool paused) {
	paused_.store(paused);
}

void RuntimeContext::waitWhilePaused(int sleepMs) const {
	if (sleepMs < 1) {
		sleepMs = 1;
	}
	while (isPaused() && !stopRectThread_.load()) {
		Sleep(static_cast<DWORD>(sleepMs));
	}
}

void RuntimeContext::refreshWindowRect(bool printHint) {
	if (!hwnd_) {
		return;
	}

	for (int attempt = 0; attempt < 4; ++attempt) {
		RECT clientRect{};
		GetClientRect(hwnd_, &clientRect);
		const int w = clientRect.right - clientRect.left;
		const int h = clientRect.bottom - clientRect.top;
		if (w <= 1 || h <= 1 || w > 9999 || h > 9999) {
			std::cout << std::endl << "No valid VRChat client area detected. Enter VRChat first." << std::endl;
			std::exit(0);
		}

		POINT p1{ clientRect.left, clientRect.top };
		POINT p2{ clientRect.right, clientRect.bottom };
		ClientToScreen(hwnd_, &p1);
		ClientToScreen(hwnd_, &p2);

		RECT screenRect{};
		screenRect.left = p1.x;
		screenRect.top = p1.y;
		screenRect.right = p2.x;
		screenRect.bottom = p2.y;

		RECT previousRect{};
		{
			std::lock_guard<std::mutex> lock(rectMutex_);
			previousRect = rect_;
			rect_ = screenRect;
		}

		const bool changed = !sameRect(screenRect, previousRect);
		if (changed && printHint) {
			std::cout << "Hint: detected window size " << w << "*" << h;
		}

		const bool forceResolution = (config_.force_resolution != 0);
		const bool needResize = forceResolution && (w != config_.target_width || h != config_.target_height);
		if (needResize) {
			if (printHint && !changed) {
				std::cout << "Hint: detected window size " << w << "*" << h;
			}
			if (printHint) {
				std::cout << ", auto resizing to " << config_.target_width << "*" << config_.target_height << std::endl;
			}

			RECT windowRect{};
			GetWindowRect(hwnd_, &windowRect);
			const int dx = (windowRect.right - windowRect.left) - (screenRect.right - screenRect.left);
			const int dy = (windowRect.bottom - windowRect.top) - (screenRect.bottom - screenRect.top);
			const int newWidth = config_.target_width + dx;
			const int newHeight = config_.target_height + dy;
			const int newLeft = (screenRect.left + screenRect.right) / 2 - newWidth / 2;
			const int newTop = (screenRect.top + screenRect.bottom) / 2 - newHeight / 2;
			MoveWindow(hwnd_, newLeft, newTop, newWidth, newHeight, TRUE);
			Sleep(80);
			continue;
		}

		if (changed && printHint) {
			std::cout << std::endl;
		}
		break;
	}
}

void RuntimeContext::refreshWindowRectLoop() {
	Sleep(2000);
	while (!stopRectThread_.load()) {
		waitWhilePaused();
		if (stopRectThread_.load()) {
			break;
		}
		refreshWindowRect(true);

		for (int i = 0; i < 10 && !stopRectThread_.load(); ++i) {
			Sleep(100);
		}
	}
}

void RuntimeContext::moveConsoleNearGameWindow() const {
	const RECT r = windowRect();
	if (r.right <= r.left || r.bottom <= r.top) {
		return;
	}
	MoveWindow(
		GetConsoleWindow(),
		r.right,
		r.top,
		(r.right - r.left) / 2,
		(r.bottom - r.top) / 2,
		TRUE);
}

}  // namespace runtime
