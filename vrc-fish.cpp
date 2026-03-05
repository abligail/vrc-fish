#include<iostream>
#include<windows.h>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <thread>
#include <fstream>
#include <vector>
#include <sstream>
#include <deque>
#include "gs-public.h"
#include "gs-opencv.h"
#include "gs-mfc.h"
#include "config/app_config.h"
#include "core/types.h"
#include "engine/controller.h"
#include "engine/detectors.h"
#include "engine/matcher.h"
#include "engine/ml_model.h"
#include "engine/template_store.h"
#include "infra/fs/path_utils.h"
#include "infra/log/logger.h"
#include "infra/win/input_api.h"
#include "infra/win/window_api.h"
#include <random>
#pragma comment(lib, "opencv_core460.lib")
#pragma comment(lib, "opencv_imgproc460.lib")
#pragma comment(lib, "opencv_imgcodecs460.lib")
#pragma comment(lib, "opencv_highgui460.lib")
#pragma comment(lib, "shell32.lib")
using namespace std;
using namespace cv;

#define KEY_PRESSED(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x0001) ? 1:0) //如果为真，表示按下过
#define KEY_PRESSING(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000) ? 1:0)  //如果为真，表示正处于按下状态

struct g_params {
	HWND hwnd{};
	RECT rect{};
	bool pause{};
	engine::TemplateStore templates;
};

// 程序全局变量
g_params params;

// 配置参数
AppConfig config;
static engine::MlpModel g_mlpModel;

static bool isPaused() {
	return params.pause;
}

static void waitWhilePaused(int sleepMs = 1000) {
	while (isPaused()) {
		Sleep(sleepMs);
	}
}

// 窗口区域初始化
void initRect() {
	HWND hwnd = params.hwnd;
	RECT rect;
	GetClientRect(hwnd, &rect);
	int w = rect.right - rect.left, h = rect.bottom - rect.top;
	if (w <= 1 || h <= 1 || w > 9999 || h > 9999) {
		std::cout << endl << "未检测到有效窗口客户区，请先进入 VRChat。" << endl;
		exit();
	}
	POINT p1 = { rect.left, rect.top };
	POINT p2 = { rect.right, rect.bottom };
	ClientToScreen(hwnd, &p1);
	ClientToScreen(hwnd, &p2);
	rect.left = p1.x, rect.top = p1.y, rect.right = p2.x, rect.bottom = p2.y;
	RECT r0 = params.rect;
	if (rect.left == r0.left && rect.right == r0.right && rect.top == r0.top && rect.bottom == r0.bottom) {
		return;
	}
	params.rect = rect;
	std::cout << "提示：捕获到窗口" << w << "*" << h;
	int targetW = config.target_width;
	int targetH = config.target_height;
	bool forceResolution = config.force_resolution != 0;
	if (forceResolution && (w != targetW || h != targetH)) {
		std::cout << "，自动为您调整为" << targetW << "*" << targetH;
		RECT windowsRect;
		GetWindowRect(hwnd, &windowsRect);
		int dx = (windowsRect.right - windowsRect.left) - (rect.right - rect.left);
		int dy = (windowsRect.bottom - windowsRect.top) - (rect.bottom - rect.top);
		int newWidth = targetW + dx, newHeight = targetH + dy;
		int newLeft = (rect.left + rect.right) / 2 - newWidth / 2;
		int newTop = (rect.top + rect.bottom) / 2 - newHeight / 2;
		MoveWindow(hwnd, newLeft, newTop, newWidth, newHeight, TRUE);
		cout << endl;
		initRect();
	}
	std::cout << endl;
}

// 窗口区域初始化 - 多线程
void initRectThread() {
	Sleep(2000);
	while (true) {
		waitWhilePaused();
		initRect();
		Sleep(1000);
	}
}

// 程序初始化
void init() {
	// 加载配置文件
	config = loadAppConfig("config.ini");

	// 获取窗口句柄
	SetConsoleTitle(L"VRChat FISH! 自动钓鱼 (Draft)");
	std::cout << "    VRChat FISH! 自动钓鱼 (Draft)" << endl << endl;
	std::cout << "  注意：本程序仅学习使用，请遵守相关规则。" << endl << endl;
	std::cout << "  提示：该模式需要使用鼠标左键按住/松开控制，小窗体建议固定为 "
		<< config.target_width << "*" << config.target_height << endl << endl;

	HWND hwnd = infra::win::findWindowByClassAndTitleContains(config.window_class, config.window_title_contains);
	if (!hwnd) {
		std::cout << "未找到 VRChat 窗口：class=" << config.window_class
			<< "，title_contains=" << config.window_title_contains << endl;
		exit();
	}

	HDC hdc = GetDC(hwnd);  // 获取窗口的设备上下文
	int dpi = GetDpiForWindow(hwnd);  // 获取窗口的 DPI 缩放比例
	ScaleViewportExtEx(hdc, dpi, dpi, dpi / 96, dpi / 96, nullptr);  // 缩放视窗区域
	ScaleWindowExtEx(hdc, dpi, dpi, dpi / 96, dpi / 96, nullptr);  // 缩放窗口区域
	params.hwnd = hwnd;

	// 窗口区域初始化
	initRect();

	// 动态识别窗口位置
	thread thInitRect(initRectThread);
	thInitRect.detach();

	// 改变脚本位置
	RECT r = params.rect;
	MoveWindow(GetConsoleWindow(), r.right, r.top, (r.right - r.left) / 2, (r.bottom - r.top) / 2, TRUE);

	params.templates = engine::loadTemplateStore(config);
}

static std::string makeDebugPath(const std::string& tag) {
	std::string dir = config.vr_debug_dir.empty() ? "debug_vrchat" : config.vr_debug_dir;
	infra::fs::ensureDirExists(dir);
	return dir + "/" + tag + "_" + std::to_string(GetTickCount64()) + ".png";
}

static void saveDebugFrame(const Mat& bgr, const std::string& tag) {
	if (!config.vr_debug_pic) {
		return;
	}
	if (bgr.empty()) {
		return;
	}
	imwrite(makeDebugPath(tag), bgr);
}

static void saveDebugFrame(const Mat& bgr, const std::string& tag, const Rect& r1, const Scalar& c1 = Scalar(0, 0, 255)) {
	if (!config.vr_debug_pic) {
		return;
	}
	if (bgr.empty()) {
		return;
	}
	Mat out = bgr.clone();
	rectangle(out, r1, c1, 2, 8, 0);
	imwrite(makeDebugPath(tag), out);
}

static void saveDebugFrame(const Mat& bgr, const std::string& tag, const Rect& r1, const Rect& r2) {
	if (!config.vr_debug_pic) {
		return;
	}
	if (bgr.empty()) {
		return;
	}
	Mat out = bgr.clone();
	rectangle(out, r1, Scalar(0, 0, 255), 2, 8, 0);
	rectangle(out, r2, Scalar(0, 255, 0), 2, 8, 0);
	imwrite(makeDebugPath(tag), out);
}

static void saveDebugFrame(const Mat& bgr, const std::string& tag, const Rect& r1, const Rect& r2, const Rect& r3) {
	if (!config.vr_debug_pic) {
		return;
	}
	if (bgr.empty()) {
		return;
	}
	Mat out = bgr.clone();
	rectangle(out, r1, Scalar(255, 0, 0), 2, 8, 0);   // 蓝框: searchRoi
	rectangle(out, r2, Scalar(0, 255, 0), 2, 8, 0);   // 绿框: 模板匹配位置
	rectangle(out, r3, Scalar(0, 0, 255), 2, 8, 0);   // 红框: 最终锁定 ROI
	imwrite(makeDebugPath(tag), out);
}

static void mouseLeftDown() {
	infra::win::mouseLeftDown(params.hwnd, params.rect, config.vr_debug);
}

static void mouseLeftUp() {
	infra::win::mouseLeftUp(params.hwnd, params.rect, config.vr_debug);
}

static void mouseLeftClickCentered(int delayMs = 40) {
	infra::win::mouseLeftClickCentered(params.hwnd, params.rect, config.vr_debug, delayMs);
}

static void mouseMoveRelative(int dx, int dy, const char* phaseTag) {
	infra::win::mouseMoveRelative(params.hwnd, params.rect, dx, dy, config.vr_debug, phaseTag);
}

static void keyTapVk(WORD vk, int delayMs = 30) {
	infra::win::keyTapVk(params.hwnd, params.rect, vk, config.vr_debug, delayMs);
}

static unsigned long long nowMs() {
	return GetTickCount64();
}

void fishVrchat() {
	VrFishState state = VrFishState::Cast;
	unsigned long long stateStart = nowMs();
	int biteOkFrames = 0;
	int minigameMissingFrames = 0;
	bool holding = false;
	bool castMouseMoved = false;
	int castMouseMoveDx = 0;
	int castMouseMoveDy = 0;
	unsigned long long lastCtrlLogMs = 0;
	int prevSliderY = 0;
	bool hasPrevSlider = false;
	double smoothVelocity = 0.0;   // EMA 平滑速度
	int prevFishY = 0;             // 上一帧鱼 Y（用于 fishVel）
	bool hasPrevFish = false;
	double smoothFishVel = 0.0;    // EMA 平滑鱼速度（px/基准帧）
	double prevSmoothFishVel = 0.0; // 上一帧的 smoothFishVel（用于算加速度）
	double smoothFishAccel = 0.0;  // EMA 平滑鱼加速度（px/基准帧^2）
	double prevDeviation = 0.0;    // 上一帧鱼与滑块中心偏差（用于方向D）
	bool hasPrevDeviation = false; // 是否有上一帧偏差
	unsigned long long prevCtrlTs = 0; // 上一帧时间戳（ms）
	bool hasPrevTs = false;
	double lastDtRatio = 1.0;      // 实际dt / 基准dt，用于MPC缩放
	double baseDtMs = config.base_dt_ms;
	if (baseDtMs < 1.0) baseDtMs = 1.0;
	int lastGoodSliderH = 0;       // 上次可信的滑块高度（用于 sH 异常时兜底）
	int lastGoodSliderCY = 0;      // 上次可信的滑块中心Y（用于 [tpl] 跳变检查）
	bool hasLastGoodPos = false;   // 是否有上次可信位置
	int consecutiveMiss = 0;       // 连续 MISS 帧数（用于 MISS 期间松开鼠标）
	Rect fixedTrackRoi{};          // 首帧定位的固定轨道 ROI（整局不变）
	bool hasFixedTrack = false;    // 是否已定位轨道

			// 快速检测缓存：首帧多尺度确定最佳缩放和模板，后续帧复用
			double cachedTrackScale = 1.0;
			double cachedTrackAngle = 0.0;
			int    cachedFishTplIdx = 0;     // params.templates.fishIcons 的索引
			bool   hasCachedFishTpl = false;

		// ML 录制模式状态
		std::ofstream recordFile;
		int recordFrame = 0;
		std::deque<int> pressWindow;   // 滑动窗口：最近 N 帧 mousePressed (0/1)
		static const int PRESS_WINDOW_SIZE = 10;

		// VRChat 日志输出文件（用于拟合物理参数，避免手动复制控制台输出）
		infra::log::Logger vrLogger;

		// ML 推理模式：加载权重
		if (config.ml_mode == 2 && !g_mlpModel.loaded) {
			if (!engine::loadMlpWeights(config.ml_weights_file, g_mlpModel)) {
				std::cerr << "[ML] 无法加载权重，退回 PD 模式" << endl;
				config.ml_mode = 0;
			}
		}

		auto writeVrLogLine = [&](const std::string& line, bool alsoStdout = true) {
			vrLogger.log(line, alsoStdout);
		};

		if (!config.vr_log_file.empty()) {
			std::string dir = infra::fs::dirNameOf(config.vr_log_file);
			if (!dir.empty()) {
				infra::fs::ensureDirExists(dir);
			}
			if (!vrLogger.openAppend(config.vr_log_file)) {
				std::cout << "[vrchat_fish] WARN: failed to open vr_log_file=" << config.vr_log_file
					<< " (check working dir / file lock)" << endl;
			} else {
				writeVrLogLine("[vrchat_fish] log start file=" + config.vr_log_file, config.vr_debug);
			}
		}

		auto switchState = [&](VrFishState next) {
			if (config.vr_debug || vrLogger.hasFile()) {
				std::ostringstream oss;
				oss << "[vrchat_fish] state " << (int)state << " -> " << (int)next;
				writeVrLogLine(oss.str(), config.vr_debug);
			}
			state = next;
			stateStart = nowMs();
		};

	auto sleepWithPause = [&](int totalMs) {
		if (totalMs <= 0) {
			return;
		}
		int remaining = totalMs;
		while (remaining > 0) {
			while (isPaused()) {
				if (holding) {
					mouseLeftUp();
					holding = false;
				}
				Sleep(1000);
			}
			int chunk = remaining > 50 ? 50 : remaining;
			Sleep(chunk);
			remaining -= chunk;
		}
	};

		auto cleanupToNextRound = [&](const std::string& tag) {
			if (holding) {
				mouseLeftUp();
				holding = false;
			}
			if (config.vr_debug || vrLogger.hasFile()) {
				std::ostringstream oss;
				oss << "[vrchat_fish] cleanup tag=" << tag
					<< " wait_before=" << config.cleanup_wait_before_ms
					<< " clicks=" << config.cleanup_click_count
					<< " click_interval=" << config.cleanup_click_interval_ms
					<< " reel_key=" << config.cleanup_reel_key_name
					<< " wait_after=" << config.cleanup_wait_after_ms;
				writeVrLogLine(oss.str(), config.vr_debug);
			}

			sleepWithPause(config.cleanup_wait_before_ms);

		int clicks = config.cleanup_click_count;
		if (clicks < 0) clicks = 0;
		int intervalMs = config.cleanup_click_interval_ms;
		if (intervalMs < 0) intervalMs = 0;
		for (int i = 0; i < clicks; i++) {
			mouseLeftClickCentered();
			if (intervalMs > 0 && i + 1 < clicks) {
				sleepWithPause(intervalMs);
			}
		}

		if (config.cleanup_reel_key > 0) {
			keyTapVk((WORD)config.cleanup_reel_key);
		}

		sleepWithPause(config.cleanup_wait_after_ms);

		// 一轮流程结束后，将鼠标移动回去（与抛竿后的偏移相反）
		if (castMouseMoved) {
			mouseMoveRelative(-castMouseMoveDx, -castMouseMoveDy, "cast_mouse_restore");
			if (config.vr_debug || vrLogger.hasFile()) {
				std::ostringstream oss;
				oss << "[vrchat_fish] cast mouse restore dx=" << -castMouseMoveDx
					<< " dy=" << -castMouseMoveDy;
				writeVrLogLine(oss.str(), config.vr_debug);
			}
			castMouseMoved = false;
			castMouseMoveDx = 0;
			castMouseMoveDy = 0;
		}
	};

	while (true) {
		while (isPaused()) {
			if (holding) {
				mouseLeftUp();
				holding = false;
			}
			Sleep(1000);
		}

		if (state == VrFishState::Cast) {
			if (holding) {
				mouseLeftUp();
				holding = false;
			}
			// 兜底：若上一轮未恢复，先恢复一次避免累积偏移
			if (castMouseMoved) {
				mouseMoveRelative(-castMouseMoveDx, -castMouseMoveDy, "cast_mouse_restore_before_cast");
				castMouseMoved = false;
				castMouseMoveDx = 0;
				castMouseMoveDy = 0;
			}
			mouseLeftClickCentered();

			if (config.cast_mouse_move_dx != 0 || config.cast_mouse_move_dy != 0) {
				// 1. 生成随机偏移后的最终位移
				static std::random_device rd;
				static std::mt19937 gen(rd());
				int range = config.cast_mouse_move_random_range;
				if (range < 0) {
					range = 0;
				}
				std::uniform_int_distribution<> dist(-range, range);
				int finalDx = config.cast_mouse_move_dx + dist(gen);
				int finalDy = config.cast_mouse_move_dy + dist(gen);

				// 2. 随机延迟（最大延迟 cast_mouse_move_delay_max）
				if (config.cast_mouse_move_delay_max > 0) {
					std::uniform_int_distribution<> delayDist(0, config.cast_mouse_move_delay_max);
					int delayMs = delayDist(gen);
					sleepWithPause(delayMs);
				}

				// 3. 平滑移动（若配置了持续时间）
				int durationMs = config.cast_mouse_move_duration_ms;
				int stepMs = config.cast_mouse_move_step_ms;
				if (durationMs > 0 && stepMs > 0) {
					// 计算步数（向上取整，确保覆盖整个持续时间）
					int steps = (durationMs + stepMs - 1) / stepMs;
					if (steps < 1) steps = 1;

					int remainingDx = finalDx;
					int remainingDy = finalDy;

					for (int i = 0; i < steps; i++) {
						// 最后一步直接移动剩余量，避免浮点累积误差
						int stepDx, stepDy;
						if (i == steps - 1) {
							stepDx = remainingDx;
							stepDy = remainingDy;
						}
						else {
							// 按时间比例分配位移（整数除法可能导致最后剩余，所以使用浮点取整）
							double ratio = (double)stepMs / durationMs;
							stepDx = (int)(finalDx * ratio);
							stepDy = (int)(finalDy * ratio);
						}

						if (stepDx != 0 || stepDy != 0) {
							mouseMoveRelative(stepDx, stepDy, "cast_mouse_move_step");
						}

						// 更新剩余位移
						remainingDx -= stepDx;
						remainingDy -= stepDy;

						// 等待步间间隔（最后一步后不再等待）
						if (i < steps - 1) {
							sleepWithPause(stepMs);
						}
					}

					// 记录最终总位移（用于后续恢复）
					castMouseMoved = true;
					castMouseMoveDx = finalDx;
					castMouseMoveDy = finalDy;
				}
				else {
					// 瞬间移动（原逻辑）
					mouseMoveRelative(finalDx, finalDy, "cast_mouse_move");
					castMouseMoved = true;
					castMouseMoveDx = finalDx;
					castMouseMoveDy = finalDy;
				}

				// 4. 日志输出
				if (config.vr_debug || vrLogger.hasFile()) {
					std::ostringstream oss;
					oss << "[vrchat_fish] cast mouse move dx=" << finalDx
						<< " dy=" << finalDy
						<< " (base: " << config.cast_mouse_move_dx
						<< "," << config.cast_mouse_move_dy
						<< " range=" << range
						<< " delay=" << config.cast_mouse_move_delay_max << "ms";
					if (durationMs > 0) {
						oss << " smooth: dur=" << durationMs << "ms step=" << stepMs << "ms";
					}
					oss << ")";
					writeVrLogLine(oss.str(), config.vr_debug);
				}
			}
			Sleep(config.cast_delay_ms);
			biteOkFrames = 0;
			switchState(VrFishState::WaitBite);
			continue;
		}

		// 点击感叹号后默认进入小游戏：仅等待固定延迟，不做"是否进入小游戏"的模板确认
		if (state == VrFishState::EnterMinigame) {
			if (nowMs() - stateStart < (unsigned long long)config.minigame_enter_delay_ms) {
				Sleep(config.capture_interval_ms);
				continue;
			}
			minigameMissingFrames = 0;
			hasPrevSlider = false;
			hasPrevFish = false;
				smoothVelocity = 0.0;
				smoothFishVel = 0.0;
				hasFixedTrack = false;
				cachedTrackScale = 1.0;
				cachedTrackAngle = 0.0;
				hasCachedFishTpl = false;
				hasPrevTs = false;
				lastDtRatio = 1.0;
				lastGoodSliderH = 0;
				lastGoodSliderCY = 0;
			hasLastGoodPos = false;
			consecutiveMiss = 0;

			if (config.ml_mode == 1) {
				// 录制模式：打开 CSV
				recordFrame = 0;
				pressWindow.clear();
				if (!recordFile.is_open()) {
					recordFile.open(config.ml_record_csv, std::ios::app);
					// 如果文件为空，写表头
					recordFile.seekp(0, std::ios::end);
					if (recordFile.tellp() == 0) {
						recordFile << "frame,timestamp_ms,fishY,sliderY,dy,sliderVel,fishVel,sliderY_norm,mousePressed,duty_label" << endl;
					}
				}
				std::cout << "[ML] 录制模式：开始录制，请手动操作鼠标控制滑块" << endl;
			}
			switchState(VrFishState::ControlMinigame);
			continue;
		}

		Mat frame = getSrc(params.rect);
		Mat gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		if (state == VrFishState::WaitBite) {
			TplMatch m{};
			bool ok = engine::detectBite(gray, params.templates, config, &m);
			if (config.vr_debug) {
				std::cout << "[vrchat_fish] bite score=" << m.score << " ok=" << ok << endl;
			}
			biteOkFrames = ok ? (biteOkFrames + 1) : 0;
			if (biteOkFrames >= config.bite_confirm_frames) {
				saveDebugFrame(frame, "bite", m.rect);
				mouseLeftClickCentered();
				hasPrevSlider = false;
				hasFixedTrack = false;
				switchState(VrFishState::EnterMinigame);
				continue;
			}
			if (nowMs() - stateStart > (unsigned long long)config.bite_timeout_ms) {
				if (config.vr_debug) {
					std::cout << "[vrchat_fish] bite timeout -> recast" << endl;
				}
				saveDebugFrame(frame, "bite_timeout");
				cleanupToNextRound("bite_timeout");
				switchState(VrFishState::Cast);
				continue;
			}
			Sleep(config.capture_interval_ms);
			continue;
		}

		if (state == VrFishState::ControlMinigame) {
			unsigned long long loopStart = nowMs();
			auto sleepControlInterval = [&]() {
				int intervalMs = config.control_interval_ms;
				if (intervalMs < 1) intervalMs = 1;
				unsigned long long elapsedMs = nowMs() - loopStart;
				if (elapsedMs < (unsigned long long)intervalMs) {
					Sleep((DWORD)((unsigned long long)intervalMs - elapsedMs));
				}
				else {
					Sleep(1);
				}
			};

				Rect searchRoi = engine::centerThirdStripRoi(gray.size());
				// ── 首帧：用 minigame_bar_full 模板定位完整轨道，取右半部分作为滑块/鱼检测 ROI ──
				if (!hasFixedTrack) {
					double barScale = 1.0;
					double barAngle = 0.0;
					TplMatch barMatch = engine::matchBestRoiTrackBarAutoScale(
						gray,
						params.templates.minigameBarFull,
						searchRoi,
						config,
						TM_CCOEFF_NORMED,
						&barScale,
						&barAngle
					);
					if (barMatch.score >= config.minigame_threshold) {
						// 轨道定位成功：取匹配区域的右半部分作为滑块和鱼的检测区域
						int trackX = barMatch.rect.x;
						int trackW = barMatch.rect.width;
						int trackY = barMatch.rect.y;
						int trackH = barMatch.rect.height;
						// 右半区域起点 = 匹配区域中点，宽度 = 一半宽度
						int halfX = trackX + trackW / 2;
						int halfW = trackW - trackW / 2;
						// 垂直方向多扩展一些（鱼可能超出轨道模板范围）
						int padY = config.track_pad_y;
						if (padY < 0) padY = 0;
						fixedTrackRoi = Rect(
							halfX,
							trackY - padY,
							halfW,
							trackH + padY * 2
						);
						fixedTrackRoi = engine::clampRect(fixedTrackRoi, gray.size());
						hasFixedTrack = true;
						cachedTrackScale = barScale;
						cachedTrackAngle = barAngle;
						// debug: 蓝框=搜索区域, 绿框=模板匹配位置, 红框=最终锁定ROI
						saveDebugFrame(frame, "track_lock", searchRoi, barMatch.rect, fixedTrackRoi);
						if (config.vr_debug || vrLogger.hasFile()) {
							std::ostringstream oss;
							oss << "[ctrl] track locked (full tpl): x=" << fixedTrackRoi.x
								<< " y=" << fixedTrackRoi.y
								<< " w=" << fixedTrackRoi.width
								<< " h=" << fixedTrackRoi.height
								<< " (bar score=" << barMatch.score
								<< " scale=" << barScale
								<< " angle=" << barAngle << ")";
							writeVrLogLine(oss.str(), config.vr_debug);
						}
					} else {
						// debug: 蓝框=搜索区域, 绿框=匹配到的(错误)位置
						saveDebugFrame(frame, "track_miss", searchRoi, barMatch.rect);
						minigameMissingFrames++;
						// 轨道长期定位不上（>= end_confirm_frames * N 帧）：放弃，退出小游戏
						int trackLockMaxMiss = config.game_end_confirm_frames * config.track_lock_miss_multiplier;
						if (trackLockMaxMiss < config.track_lock_miss_min_frames) trackLockMaxMiss = config.track_lock_miss_min_frames;
						if (config.vr_debug || vrLogger.hasFile()) {
							std::ostringstream oss;
							oss << "[ctrl] track detect MISS (score=" << barMatch.score
								<< " scale=" << barScale
								<< " angle=" << barAngle
								<< ") miss=" << minigameMissingFrames << "/" << trackLockMaxMiss;
							writeVrLogLine(oss.str(), config.vr_debug);
						}
						if (minigameMissingFrames >= trackLockMaxMiss) {
							if (holding) { mouseLeftUp(); holding = false; }
							saveDebugFrame(frame, "track_lock_timeout", searchRoi);
							switchState(VrFishState::PostMinigame);
						}
						sleepControlInterval();
						continue;
					}
				}
	
				Rect matchRoi = fixedTrackRoi;
	
				FishSliderResult det;
				bool ok = false;
				bool didFullDetect = false;
				// 首帧：在已锁定轨道的 scale/angle 下，选择最匹配的鱼模板索引
				// 后续帧：固定 scale/angle + 单模板快速检测
				// 只在快速路径失败时才回退到全检测（可能鱼图标模板变了）
				if (!hasCachedFishTpl) {
					int bestIdx = 0;
					ok = engine::detectFishAndSliderFull(
						gray, matchRoi, params.templates, config, cachedTrackScale, cachedTrackAngle, &det, &bestIdx);
					if (ok) {
						cachedFishTplIdx = bestIdx;
						hasCachedFishTpl = true;
					}
					didFullDetect = true;
				} else {
					ok = engine::detectFishAndSliderFast(
						gray, matchRoi, params.templates, config, cachedTrackScale, cachedTrackAngle, cachedFishTplIdx, &det);
					if (!ok) {
						int bestIdx = 0;
						ok = engine::detectFishAndSliderFull(
							gray, matchRoi, params.templates, config, cachedTrackScale, cachedTrackAngle, &det, &bestIdx);
						if (ok) {
							cachedFishTplIdx = bestIdx;
						}
						didFullDetect = true;
					}
				}
	
				unsigned long long detectMs = nowMs() - loopStart;

			if (!ok) {
				if (config.vr_debug || vrLogger.hasFile()) {
					std::ostringstream oss;
					oss << "[ctrl] " << detectMs << "ms"
						<< (didFullDetect ? " [full]" : " [fast]")
						<< " MISS fs=" << det.fishScore
						<< " ss=" << det.sliderScore
						<< " hold=" << (holding ? 1 : 0);
					writeVrLogLine(oss.str(), config.vr_debug);
				}
				minigameMissingFrames++;
				consecutiveMiss++;
				// 连续 MISS >= N 帧才松开鼠标（避免单次全检测失败导致松手）
				int missReleaseFrames = config.miss_release_frames;
				if (missReleaseFrames < 1) missReleaseFrames = 1;
				if (consecutiveMiss >= missReleaseFrames && holding) {
					mouseLeftUp();
					holding = false;
				}
				int endFrames = config.game_end_confirm_frames;
				if (endFrames < config.minigame_end_min_frames) endFrames = config.minigame_end_min_frames;
				if (minigameMissingFrames >= endFrames) {
					if (holding) {
						mouseLeftUp();
						holding = false;
					}
					saveDebugFrame(frame, "minigame_end", fixedTrackRoi);
					switchState(VrFishState::PostMinigame);
				}
				sleepControlInterval();
				continue;
			}

			// MISS 恢复后重置速度估算（MISS 期间位置跳变，速度不可信）
			bool wasLongMiss = (consecutiveMiss >= 2);
			minigameMissingFrames = 0;
			consecutiveMiss = 0;

			// ── [tpl] 位置跳变检查：模板匹配位置不可信时用上次好值 ──
			if (!det.hasBounds && hasLastGoodPos) {
				// 颜色检测失败（[tpl]），检查 sCY 是否跳变太大
				int scyJump = abs(det.sliderCenterY - lastGoodSliderCY);
				if (scyJump > config.slider_tpl_jump_threshold) {
					// sCY 跳变过大，用上次好位置替代
					det.sliderCenterY = lastGoodSliderCY;
					det.sliderTop = lastGoodSliderCY - lastGoodSliderH / 2;
					det.sliderBottom = lastGoodSliderCY + lastGoodSliderH / 2;
					det.sliderHeight = lastGoodSliderH;
				}
			}

			// ── fishY 跳变保护：鱼移动缓慢，单帧大跳变必为检测错误 ──
			if (hasPrevFish) {
				int fishJump = abs(det.fishY - prevFishY);
				if (fishJump > config.fish_jump_threshold) {
					det.fishY = prevFishY; // 用上一帧值替代
				}
			}

			// ── 滑块高度修正：颜色检测不稳定时用历史值兜底 ──
			if (det.sliderHeight >= config.slider_height_stable_min) {
				lastGoodSliderH = det.sliderHeight;
			} else if (lastGoodSliderH > 0) {
				det.sliderHeight = lastGoodSliderH;
				det.sliderTop = det.sliderCenterY - lastGoodSliderH / 2;
				det.sliderBottom = det.sliderCenterY + lastGoodSliderH / 2;
			}

			// 记录可信位置（仅 [color] 检测且位置合理时更新）
			if (det.hasBounds) {
				lastGoodSliderCY = det.sliderCenterY;
				hasLastGoodPos = true;
			}

			// trackRoi 已被 fixedTrackRoi 取代，无需动态更新

			// ── 计算速度特征（时间归一化到基准帧间隔） ──
				{
					int fishY = det.fishY;
					int sliderCY = det.sliderCenterY;
					int sliderH = det.sliderHeight;

					unsigned long long t = nowMs();

					// 计算实际 dt 和时间缩放比
					double dtMs = baseDtMs;
					if (hasPrevTs && t > prevCtrlTs) {
						dtMs = (double)(t - prevCtrlTs);
						if (dtMs < 1.0) dtMs = 1.0;
						if (dtMs > 1000.0) dtMs = 1000.0; // cap at 1s
					}

					// 输出 ctrl 日志（包含 dt，便于离线拟合按真实 dt 归一化）
					if (config.vr_debug || vrLogger.hasFile()) {
						std::ostringstream oss;
						oss << "[ctrl] " << detectMs << "ms"
							<< (didFullDetect ? " [full]" : " [fast]")
							<< " dt=" << (int)dtMs << "ms"
							<< " t=" << t
							<< " fishY=" << fishY
							<< " sCY=" << sliderCY
							<< " sH=" << sliderH
							<< (det.hasBounds ? " [color]" : " [tpl]")
							<< " hold=" << (holding ? 1 : 0);
						writeVrLogLine(oss.str(), config.vr_debug);
					}

					lastDtRatio = dtMs / baseDtMs;
					prevCtrlTs = t;
					hasPrevTs = true;

				// EMA 平滑 slider 速度（归一化到 px/基准帧）
				double alpha = config.velocity_ema_alpha;
				if (alpha < 0.05) alpha = 0.05;
				if (alpha > 1.0) alpha = 1.0;

				// 位置大跳变：速度估算直接作废（避免 [tpl]↔[color] 切换导致 sv 饱和）
				if (hasPrevSlider) {
					int jumpThresh = config.slider_tpl_jump_threshold;
					if (jumpThresh < 50) jumpThresh = 50;
					if (abs(sliderCY - prevSliderY) > jumpThresh) {
						hasPrevSlider = false;
						smoothVelocity = 0.0;
					}
				}

				// MISS 恢复后或 dt 过大时：衰减速度（完全归零会让 MPC 误判）
				if (wasLongMiss || dtMs > 300.0) {
					// 按 dt 衰减：dt 越长衰减越多，但保留方向信息
					double decayFactor = 0.3; // 保留 30% 速度
					if (dtMs > 500.0) decayFactor = 0.1;
					if (dtMs > 800.0) decayFactor = 0.0;
					smoothVelocity *= decayFactor;
					smoothFishVel *= decayFactor;
					smoothFishAccel *= decayFactor;
					prevSmoothFishVel = smoothFishVel;
					hasPrevDeviation = false;
					hasPrevSlider = false;
					hasPrevFish = false;
				}

				// 原始位移归一化到基准帧间隔
				double rawV = hasPrevSlider ? (double)(sliderCY - prevSliderY) / lastDtRatio : 0.0;
				if (!hasPrevSlider) {
					smoothVelocity = 0.0;
				} else {
					smoothVelocity = alpha * rawV + (1.0 - alpha) * smoothVelocity;
					// 钳制速度到合理范围（防止检测跳变导致极端值）
					double maxVel = config.slider_velocity_cap;
					if (maxVel < 1.0) maxVel = 1.0;
					if (smoothVelocity > maxVel) smoothVelocity = maxVel;
					if (smoothVelocity < -maxVel) smoothVelocity = -maxVel;
				}

				// EMA 平滑 fish 速度（归一化到 px/基准帧）
				double rawFV = hasPrevFish ? (double)(fishY - prevFishY) / lastDtRatio : 0.0;
				if (!hasPrevFish) {
					smoothFishVel = 0.0;
				} else {
					smoothFishVel = alpha * rawFV + (1.0 - alpha) * smoothFishVel;
					double fishVelCap = config.fish_velocity_cap;
					if (fishVelCap < 1.0) fishVelCap = 1.0;
					if (smoothFishVel > fishVelCap) smoothFishVel = fishVelCap;
					if (smoothFishVel < -fishVelCap) smoothFishVel = -fishVelCap;
				}

				// ── 方向A：鱼加速度 EMA 追踪 ──
				{
					double accelAlpha = config.fish_accel_alpha;
					if (accelAlpha < 0.05) accelAlpha = 0.05;
					if (accelAlpha > 1.0) accelAlpha = 1.0;
					double rawAccel = smoothFishVel - prevSmoothFishVel;
					smoothFishAccel = accelAlpha * rawAccel + (1.0 - accelAlpha) * smoothFishAccel;
					double accelCap = config.fish_accel_cap;
					if (accelCap < 0.5) accelCap = 0.5;
					if (smoothFishAccel > accelCap) smoothFishAccel = accelCap;
					if (smoothFishAccel < -accelCap) smoothFishAccel = -accelCap;
					prevSmoothFishVel = smoothFishVel;
				}

				if (config.ml_mode == 1) {
					// ══ 录制模式：只读鼠标状态，写 CSV ══
					int dy = sliderCY - fishY;
					double sliderYNorm = (gray.rows > 0) ? (double)sliderCY / gray.rows : 0.5;
					int mousePressed = KEY_PRESSING(VK_LBUTTON) ? 1 : 0;
					pressWindow.push_back(mousePressed);
					if ((int)pressWindow.size() > PRESS_WINDOW_SIZE)
						pressWindow.pop_front();

					double dutyLabel = -1.0;
					if ((int)pressWindow.size() >= PRESS_WINDOW_SIZE) {
						int sum = 0;
						for (int v : pressWindow) sum += v;
						dutyLabel = (double)sum / PRESS_WINDOW_SIZE;
					}

					if (recordFile.is_open()) {
						recordFile << recordFrame << ","
							<< t << ","
							<< fishY << ","
							<< sliderCY << ","
							<< dy << ","
							<< smoothVelocity << ","
							<< smoothFishVel << ","
							<< sliderYNorm << ","
							<< mousePressed << ","
							<< dutyLabel << endl;
					}
					recordFrame++;

					if (config.vr_debug && (t - lastCtrlLogMs >= 500)) {
						std::cout << "[ML:record] frame=" << recordFrame
							<< " dy=" << dy
							<< " sv=" << (int)smoothVelocity
							<< " fv=" << (int)smoothFishVel
							<< " sH=" << sliderH
							<< " mouse=" << mousePressed
							<< endl;
						lastCtrlLogMs = t;
					}
				} else {
					engine::ControlInput controlInput{};
					controlInput.fishY = fishY;
					controlInput.sliderCenterY = sliderCY;
					controlInput.sliderHeight = sliderH;
					controlInput.smoothSliderVelocity = smoothVelocity;
					controlInput.smoothFishVelocity = smoothFishVel;
					controlInput.smoothFishAccel = smoothFishAccel;
					controlInput.holding = holding;
					controlInput.fixedTrackRoi = fixedTrackRoi;
					controlInput.lastDtRatio = lastDtRatio;
					controlInput.prevDeviation = prevDeviation;
					controlInput.hasPrevDeviation = hasPrevDeviation;

					engine::ControlDecision decision = engine::computeControlDecision(controlInput, config);
					bool wantHold = decision.wantHold;
					bool reactiveTriggered = decision.reactiveTriggered;
					double costPress = decision.costPress;
					double costRelease = decision.costRelease;
					prevDeviation = decision.deviationForNext;
					hasPrevDeviation = decision.hasDeviationForNext;

					if (wantHold && !holding) {
						mouseLeftDown();
						holding = true;
					} else if (!wantHold && holding) {
						mouseLeftUp();
						holding = false;
					}

					if (config.vr_debug || vrLogger.hasFile()) {
						int logIntervalMs = config.bb_log_interval_ms;
						if (logIntervalMs < 0) logIntervalMs = 0;
						if (logIntervalMs == 0 || t - lastCtrlLogMs >= (unsigned long long)logIntervalMs) {
							std::ostringstream oss;
							oss << "[MPC] dt=" << (int)(lastDtRatio * baseDtMs) << "ms"
								<< " fishY=" << fishY
								<< " sCY=" << sliderCY
								<< " sH=" << sliderH
								<< " sv=" << (int)smoothVelocity
								<< " fv=" << (int)smoothFishVel
								<< " fa=" << (int)smoothFishAccel
								<< " cP=" << (int)costPress
								<< " cR=" << (int)costRelease
								<< " hold=" << (holding ? 1 : 0)
								<< (reactiveTriggered ? " [reactive]" : "");
							writeVrLogLine(oss.str(), config.vr_debug);
							lastCtrlLogMs = t;
						}
					}
					}

				prevSliderY = sliderCY;
				hasPrevSlider = true;
				prevFishY = fishY;
				hasPrevFish = true;
			}

			sleepControlInterval();
			continue;
		}

		if (state == VrFishState::PostMinigame) {
			saveDebugFrame(frame, "post_minigame");
			// 录制模式：flush CSV
			if (config.ml_mode == 1 && recordFile.is_open()) {
				recordFile.flush();
				std::cout << "[ML] 本轮录制完成，已写入 " << recordFrame << " 帧" << endl;
			}
			// 回合结束后不再依赖 xp.png 识别：固定等待 -> 多次点击入包 -> 按 T 强制收杆 -> 再等待 -> 下一轮
			cleanupToNextRound("post_minigame");
			switchState(VrFishState::Cast);
			continue;
		}
	}
}

int main() {

	init();
	std::cout << endl;

	if (config.is_pause) {
		std::cout << "按下Tab键可暂停/继续" << endl;
		thread th(fishVrchat);
		th.detach();
		while (1) {
			if (KEY_PRESSED(VK_TAB)) {
				if (params.pause) {
					params.pause = false;
					std::cout << "已继续" << endl;
				}
				else {
					params.pause = true;
					std::cout << "已暂停" << endl;
				}
			}
			Sleep(500);
		}
	}
	else {
		fishVrchat();
	}
	return 0;
}
