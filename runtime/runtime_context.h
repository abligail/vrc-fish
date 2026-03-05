#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include <windows.h>
#include <opencv2/core/mat.hpp>

#include "../config/app_config.h"
#include "../engine/template_store.h"

namespace runtime {

class RuntimeContext {
public:
	RuntimeContext();
	~RuntimeContext();

	bool initialize(const std::string& configPath = "config.ini");
	void shutdown();

	AppConfig& config();
	const AppConfig& config() const;

	const engine::TemplateStore& templates() const;

	HWND hwnd() const;
	RECT windowRect() const;
	cv::Mat captureBgr() const;

	void mouseLeftDown() const;
	void mouseLeftUp() const;
	void mouseLeftClickCentered(int delayMs = 40) const;
	void mouseMoveRelative(int dx, int dy, const char* phaseTag) const;
	void keyTapVk(WORD vk, int delayMs = 30) const;

	bool isPaused() const;
	bool togglePause();
	void setPaused(bool paused);
	void waitWhilePaused(int sleepMs = 1000) const;

private:
	void refreshWindowRect(bool printHint);
	void refreshWindowRectLoop();
	void moveConsoleNearGameWindow() const;

	AppConfig config_{};
	HWND hwnd_{};
	RECT rect_{};
	engine::TemplateStore templates_{};

	std::atomic<bool> paused_{ false };
	std::atomic<bool> stopRectThread_{ false };
	std::thread rectThread_{};
	mutable std::mutex rectMutex_{};
};

}  // namespace runtime
