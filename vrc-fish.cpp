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
#include <ShlObj.h>
#include "ini.h"
#include "gs-public.h"
#include "gs-opencv.h"
#include "gs-mfc.h"
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
	struct GrayTpl {
		Mat gray;
		Mat mask;  // 非空时启用 masked matchTemplate
		bool empty() const { return gray.empty(); }
		int cols() const { return gray.cols; }
		int rows() const { return gray.rows; }
		Size size() const { return gray.size(); }
	};

	HWND hwnd{};
	RECT rect{};
	bool pause{};

	GrayTpl tpl_vr_bite_excl_bottom;
	GrayTpl tpl_vr_bite_excl_full;
	GrayTpl tpl_vr_minigame_bar_full;
	GrayTpl tpl_vr_fish_icon;
	GrayTpl tpl_vr_fish_icon_alt;
	GrayTpl tpl_vr_fish_icon_alt2;
	std::vector<GrayTpl> tpl_vr_fish_icons;           // fish_icon + fish_icon_alt*.png（动态加载）
	std::vector<std::string> tpl_vr_fish_icon_files;  // 与 tpl_vr_fish_icons 对应的文件名（用于日志/调试）
	GrayTpl tpl_vr_player_slider;
};

struct g_config {
	bool is_pause;

	string window_class;
	string window_title_contains;
	int force_resolution;
	int target_width;
	int target_height;
	int capture_interval_ms;
	int cast_delay_ms;
	int cast_mouse_move_dx;
	int cast_mouse_move_dy;
	int cast_mouse_move_random_range;
	int cast_mouse_move_delay_max;
	int cast_mouse_move_duration_ms;   // 新增：移动持续时间（0=瞬间）
	int cast_mouse_move_step_ms;       // 新增：步间间隔
	int bite_timeout_ms;
	int minigame_enter_delay_ms;
	int cleanup_wait_before_ms;
	int cleanup_click_count;
	int cleanup_click_interval_ms;
	string cleanup_reel_key_name;
	int cleanup_reel_key;
	int cleanup_wait_after_ms;
	int bite_confirm_frames;
	int game_end_confirm_frames;
	double bite_threshold;
	double minigame_threshold;
	double fish_icon_threshold;
	double slider_threshold;
	int control_interval_ms;
	double velocity_ema_alpha;
	int slider_bright_thresh;
	int slider_min_height;
	double bb_gravity;
	double bb_thrust;
	double bb_drag;
	int bb_sim_horizon;
	double bb_margin_ratio;
	double bb_boundary_zone;
	double bb_boundary_weight;
	int bb_log_interval_ms;
	int ml_mode;
	string ml_record_csv;
	string ml_weights_file;

	string resource_dir;
	string tpl_bite_exclamation_bottom;
	string tpl_bite_exclamation_full;
	string tpl_minigame_bar_full;
	string tpl_fish_icon;
	string tpl_fish_icon_alt;
	string tpl_fish_icon_alt2;
	string tpl_player_slider;

	double fish_scale_1;
	double fish_scale_2;
	double fish_scale_3;
	double fish_scale_4;
	double track_scale_1;
	double track_scale_2;
	double track_scale_3;
	double track_scale_4;
	// 轨道模板自动多尺度（优先级高于 track_scale_1~4）
	double track_scale_min;
	double track_scale_max;
	double track_scale_step;
	int track_scale_refine_topk;
	double track_scale_refine_radius;
	double track_scale_refine_step;
	// 轨道模板旋转角度搜索（单位：度）
	double track_angle_min;
	double track_angle_max;
	double track_angle_step;
	int track_angle_refine_topk;
	double track_angle_refine_radius;
	double track_angle_refine_step;

	int slider_detect_half_width;
	int slider_detect_merge_gap;
	int track_pad_y;
	int track_lock_miss_multiplier;
	int track_lock_miss_min_frames;
	int miss_release_frames;
	int minigame_end_min_frames;
	int slider_tpl_jump_threshold;
	int fish_jump_threshold;
	int slider_height_stable_min;
	double slider_velocity_cap;
	double fish_velocity_cap;
	double base_dt_ms;

	// 鱼预测增强（方向A）
	int fish_bounce_predict;        // 1=MPC中鱼碰轨道边界会反弹
	double fish_accel_alpha;        // 鱼加速度EMA平滑系数（0~1）
	double fish_vel_decay;          // MPC中鱼速度逐步衰减系数（0~1，1=不衰减）
	double fish_accel_cap;          // 鱼加速度上限（px/基准帧^2）

	// 偏差快速响应（方向D）
	int reactive_override;          // 1=启用偏差快速响应
	double reactive_dev_ratio;      // 触发阈值：偏差 > sliderH * ratio 时触发
	double reactive_grow_threshold; // 偏差增长速度阈值（px/基准帧）

	bool vr_debug;
	bool vr_debug_pic;
	string vr_debug_dir;
	string vr_log_file;
};

// 程序全局变量
g_params params;

// 配置参数
g_config config;

static bool isPaused() {
	return params.pause;
}

static void waitWhilePaused(int sleepMs = 1000) {
	while (isPaused()) {
		Sleep(sleepMs);
	}
}

// 读取配置文件
void loadConfig() {
	ZIni ini("config.ini");
	config.is_pause = ini.getInt("common", "is_pause", 1);

	config.window_class = ini.get("vrchat_fish", "window_class", "UnityWndClass");
	config.window_title_contains = ini.get("vrchat_fish", "window_title_contains", "VRChat");
	config.force_resolution = ini.getInt("vrchat_fish", "force_resolution", 1);
	config.target_width = ini.getInt("vrchat_fish", "target_width", 1280);
	config.target_height = ini.getInt("vrchat_fish", "target_height", 960);
	config.capture_interval_ms = ini.getInt("vrchat_fish", "capture_interval_ms", 80);
	config.cast_delay_ms = ini.getInt("vrchat_fish", "cast_delay_ms", 200);
	config.cast_mouse_move_dx = ini.getInt("vrchat_fish", "cast_mouse_move_dx", 0);
	config.cast_mouse_move_dy = ini.getInt("vrchat_fish", "cast_mouse_move_dy", 0);
	config.cast_mouse_move_random_range = ini.getInt("vrchat_fish", "cast_mouse_move_random_range", 0);
	config.cast_mouse_move_delay_max = ini.getInt("vrchat_fish", "cast_mouse_move_delay_max", 0);
	config.cast_mouse_move_duration_ms = ini.getInt("vrchat_fish", "cast_mouse_move_duration_ms", 0);
	config.cast_mouse_move_step_ms = ini.getInt("vrchat_fish", "cast_mouse_move_step_ms", 30);
	config.bite_timeout_ms = ini.getInt("vrchat_fish", "bite_timeout_ms", 12000);
	config.minigame_enter_delay_ms = ini.getInt("vrchat_fish", "minigame_enter_delay_ms", 150);
	config.cleanup_wait_before_ms = ini.getInt("vrchat_fish", "cleanup_wait_before_ms", 800);
	config.cleanup_click_count = ini.getInt("vrchat_fish", "cleanup_click_count", 4);
	config.cleanup_click_interval_ms = ini.getInt("vrchat_fish", "cleanup_click_interval_ms", 80);
	config.cleanup_reel_key_name = ini.get("vrchat_fish", "cleanup_reel_key", "T");
	config.cleanup_reel_key = getKeyVal(config.cleanup_reel_key_name);
	int legacyLootClickDelayMs = ini.getInt("vrchat_fish", "loot_click_delay_ms", 200);
	config.cleanup_wait_after_ms = ini.getInt("vrchat_fish", "cleanup_wait_after_ms", legacyLootClickDelayMs);
	config.bite_confirm_frames = ini.getInt("vrchat_fish", "bite_confirm_frames", 3);
	config.game_end_confirm_frames = ini.getInt("vrchat_fish", "game_end_confirm_frames", 3);
	config.bite_threshold = ini.getDouble("vrchat_fish", "bite_threshold", 0.80);
	config.minigame_threshold = ini.getDouble("vrchat_fish", "minigame_threshold", 0.75);
	config.fish_icon_threshold = ini.getDouble("vrchat_fish", "fish_icon_threshold", 0.75);
	config.slider_threshold = ini.getDouble("vrchat_fish", "slider_threshold", 0.75);
	config.control_interval_ms = ini.getInt("vrchat_fish", "control_interval_ms", 30);
	config.velocity_ema_alpha = ini.getDouble("vrchat_fish", "velocity_ema_alpha", 0.3);
	config.slider_bright_thresh = ini.getInt("vrchat_fish", "slider_bright_thresh", 180);
	config.slider_min_height = ini.getInt("vrchat_fish", "slider_min_height", 15);
	config.bb_gravity = ini.getDouble("vrchat_fish", "bb_gravity", 2.0);
	config.bb_thrust = ini.getDouble("vrchat_fish", "bb_thrust", -2.0);
	config.bb_drag = ini.getDouble("vrchat_fish", "bb_drag", 0.85);
	config.bb_sim_horizon = ini.getInt("vrchat_fish", "bb_sim_horizon", 8);
	config.bb_margin_ratio = ini.getDouble("vrchat_fish", "bb_margin_ratio", 0.25);
	config.bb_boundary_zone = ini.getDouble("vrchat_fish", "bb_boundary_zone", 40.0);
	config.bb_boundary_weight = ini.getDouble("vrchat_fish", "bb_boundary_weight", 0.3);
	config.bb_log_interval_ms = ini.getInt("vrchat_fish", "bb_log_interval_ms", 200);

	config.ml_mode = ini.getInt("vrchat_fish", "ml_mode", 0);
	config.ml_record_csv = ini.get("vrchat_fish", "ml_record_csv", "record_data.csv");
	config.ml_weights_file = ini.get("vrchat_fish", "ml_weights_file", "ml_weights.txt");

	config.resource_dir = ini.get("vrchat_fish", "resource_dir", "Resource-VRChat");
	config.tpl_bite_exclamation_bottom = ini.get("vrchat_fish", "tpl_bite_exclamation_bottom", "bite_exclamation_bottom.png");
	config.tpl_bite_exclamation_full = ini.get("vrchat_fish", "tpl_bite_exclamation_full", "bite_exclamation_full.png");
	config.tpl_minigame_bar_full = ini.get("vrchat_fish", "tpl_minigame_bar_full", "minigame_bar_full.png");
	config.tpl_fish_icon = ini.get("vrchat_fish", "tpl_fish_icon", "fish_icon.png");
	config.tpl_fish_icon_alt = ini.get("vrchat_fish", "tpl_fish_icon_alt", "fish_icon_alt.png");
	config.tpl_fish_icon_alt2 = ini.get("vrchat_fish", "tpl_fish_icon_alt2", "fish_icon_alt2.png");
	config.tpl_player_slider = ini.get("vrchat_fish", "tpl_player_slider", "player_slider.png");

	config.fish_scale_1 = ini.getDouble("vrchat_fish", "fish_scale_1", 0.9);
	config.fish_scale_2 = ini.getDouble("vrchat_fish", "fish_scale_2", 1.0);
	config.fish_scale_3 = ini.getDouble("vrchat_fish", "fish_scale_3", 1.2);
	config.fish_scale_4 = ini.getDouble("vrchat_fish", "fish_scale_4", 1.5);
	config.track_scale_1 = ini.getDouble("vrchat_fish", "track_scale_1", 0.9);
	config.track_scale_2 = ini.getDouble("vrchat_fish", "track_scale_2", 1.0);
	config.track_scale_3 = ini.getDouble("vrchat_fish", "track_scale_3", 1.2);
	config.track_scale_4 = ini.getDouble("vrchat_fish", "track_scale_4", 1.5);
	config.track_scale_min = ini.getDouble("vrchat_fish", "track_scale_min", 0.9);
	config.track_scale_max = ini.getDouble("vrchat_fish", "track_scale_max", 2.0);
	config.track_scale_step = ini.getDouble("vrchat_fish", "track_scale_step", 0.1);
	config.track_scale_refine_topk = ini.getInt("vrchat_fish", "track_scale_refine_topk", 2);
	config.track_scale_refine_radius = ini.getDouble("vrchat_fish", "track_scale_refine_radius", 0.05);
	config.track_scale_refine_step = ini.getDouble("vrchat_fish", "track_scale_refine_step", 0.01);
	config.track_angle_min = ini.getDouble("vrchat_fish", "track_angle_min", -2.0);
	config.track_angle_max = ini.getDouble("vrchat_fish", "track_angle_max", 2.0);
	config.track_angle_step = ini.getDouble("vrchat_fish", "track_angle_step", 0.2);
	config.track_angle_refine_topk = ini.getInt("vrchat_fish", "track_angle_refine_topk", 2);
	config.track_angle_refine_radius = ini.getDouble("vrchat_fish", "track_angle_refine_radius", 0.2);
	config.track_angle_refine_step = ini.getDouble("vrchat_fish", "track_angle_refine_step", 0.05);
	config.slider_detect_half_width = ini.getInt("vrchat_fish", "slider_detect_half_width", 4);
	config.slider_detect_merge_gap = ini.getInt("vrchat_fish", "slider_detect_merge_gap", 40);
	config.track_pad_y = ini.getInt("vrchat_fish", "track_pad_y", 30);
	config.track_lock_miss_multiplier = ini.getInt("vrchat_fish", "track_lock_miss_multiplier", 6);
	config.track_lock_miss_min_frames = ini.getInt("vrchat_fish", "track_lock_miss_min_frames", 30);
	config.miss_release_frames = ini.getInt("vrchat_fish", "miss_release_frames", 3);
	config.minigame_end_min_frames = ini.getInt("vrchat_fish", "minigame_end_min_frames", 10);
	config.slider_tpl_jump_threshold = ini.getInt("vrchat_fish", "slider_tpl_jump_threshold", 150);
	config.fish_jump_threshold = ini.getInt("vrchat_fish", "fish_jump_threshold", 80);
	config.slider_height_stable_min = ini.getInt("vrchat_fish", "slider_height_stable_min", 80);
	config.slider_velocity_cap = ini.getDouble("vrchat_fish", "slider_velocity_cap", 30.0);
	config.fish_velocity_cap = ini.getDouble("vrchat_fish", "fish_velocity_cap", 15.0);
	config.base_dt_ms = ini.getDouble("vrchat_fish", "base_dt_ms", 50.0);

	config.fish_bounce_predict = ini.getInt("vrchat_fish", "fish_bounce_predict", 1);
	config.fish_accel_alpha = ini.getDouble("vrchat_fish", "fish_accel_alpha", 0.4);
	config.fish_vel_decay = ini.getDouble("vrchat_fish", "fish_vel_decay", 0.92);
	config.fish_accel_cap = ini.getDouble("vrchat_fish", "fish_accel_cap", 5.0);

	config.reactive_override = ini.getInt("vrchat_fish", "reactive_override", 1);
	config.reactive_dev_ratio = ini.getDouble("vrchat_fish", "reactive_dev_ratio", 0.55);
	config.reactive_grow_threshold = ini.getDouble("vrchat_fish", "reactive_grow_threshold", 3.0);

	config.vr_debug = ini.getInt("vrchat_fish", "debug", 1);
	config.vr_debug_pic = ini.getInt("vrchat_fish", "debug_pic", 0);
	config.vr_debug_dir = ini.get("vrchat_fish", "debug_dir", "debug_vrchat");
	config.vr_log_file = ini.get("vrchat_fish", "vr_log_file", "");

	std::cout << "已加载 VRChat 配置: window_class=" << config.window_class
		<< ", title_contains=" << config.window_title_contains
		<< ", target=" << config.target_width << "*" << config.target_height << endl;
	if (config.ml_mode == 1) {
		std::cout << "[ML] 录制模式已启用，CSV输出: " << config.ml_record_csv << endl;
	} else if (config.ml_mode == 2) {
		std::cout << "[ML] 推理模式已启用，权重文件: " << config.ml_weights_file << endl;
	}
}

static std::wstring toWStringSimple(const std::string& s) {
	return std::wstring(s.begin(), s.end());
}

struct FindWindowCtx {
	std::wstring className;
	std::wstring titleContains;
	HWND hwnd = nullptr;
};

static BOOL CALLBACK enumWindowsFindProc(HWND hwnd, LPARAM lParam) {
	auto* ctx = reinterpret_cast<FindWindowCtx*>(lParam);
	if (!IsWindowVisible(hwnd)) {
		return TRUE;
	}
	if (!ctx->className.empty()) {
		wchar_t cls[256]{};
		GetClassNameW(hwnd, cls, 256);
		if (ctx->className != cls) {
			return TRUE;
		}
	}
	if (!ctx->titleContains.empty()) {
		wchar_t title[512]{};
		GetWindowTextW(hwnd, title, 512);
		std::wstring t = title;
		if (t.find(ctx->titleContains) == std::wstring::npos) {
			return TRUE;
		}
	}
	ctx->hwnd = hwnd;
	return FALSE;
}

HWND findWindowByClassAndTitleContains(const std::string& windowClass, const std::string& titleContains) {
	FindWindowCtx ctx{};
	ctx.className = toWStringSimple(windowClass);
	ctx.titleContains = toWStringSimple(titleContains);
	EnumWindows(enumWindowsFindProc, reinterpret_cast<LPARAM>(&ctx));
	return ctx.hwnd;
}

g_params::GrayTpl loadGrayTplFromFile(const std::string& path) {
	Mat raw = imread(path, IMREAD_UNCHANGED);  // 保留 alpha 通道
	if (raw.empty()) {
		std::cout << "模板加载失败：" << path << endl;
		exit();
	}
	g_params::GrayTpl tpl{};
	if (raw.channels() == 4) {
		// 分离 alpha 通道作为 mask
		std::vector<Mat> channels;
		split(raw, channels);
		Mat alpha = channels[3];
		// 如果 alpha 不全是 255，说明有透明区域，启用 mask
		double minA = 0, maxA = 0;
		minMaxLoc(alpha, &minA, &maxA);
		Mat bgr;
		cvtColor(raw, bgr, COLOR_BGRA2BGR);
		cvtColor(bgr, tpl.gray, COLOR_BGR2GRAY);
		if (minA < 255.0) {
			// 二值化 mask：alpha > 0 -> 255, 否则 0
			threshold(alpha, tpl.mask, 0, 255, THRESH_BINARY);
			std::cout << "模板已加载（带mask）：" << path << endl;
		} else {
			std::cout << "模板已加载：" << path << endl;
		}
	} else {
		Mat bgr = raw;
		if (raw.channels() == 1) {
			tpl.gray = raw;
		} else {
			cvtColor(bgr, tpl.gray, COLOR_BGR2GRAY);
		}
		std::cout << "模板已加载：" << path << endl;
	}
	return tpl;
}

static g_params::GrayTpl tryLoadGrayTplFromFile(const std::string& path) {
	Mat raw = imread(path, IMREAD_UNCHANGED);  // 保留 alpha 通道
	if (raw.empty()) {
		std::cout << "模板加载失败（忽略）：" << path << endl;
		return g_params::GrayTpl{};
	}
	g_params::GrayTpl tpl{};
	if (raw.channels() == 4) {
		// 分离 alpha 通道作为 mask
		std::vector<Mat> channels;
		split(raw, channels);
		Mat alpha = channels[3];
		// 如果 alpha 不全是 255，说明有透明区域，启用 mask
		double minA = 0, maxA = 0;
		minMaxLoc(alpha, &minA, &maxA);
		Mat bgr;
		cvtColor(raw, bgr, COLOR_BGRA2BGR);
		cvtColor(bgr, tpl.gray, COLOR_BGR2GRAY);
		if (minA < 255.0) {
			// 二值化 mask：alpha > 0 -> 255, 否则 0
			threshold(alpha, tpl.mask, 0, 255, THRESH_BINARY);
			std::cout << "模板已加载（带mask）：" << path << endl;
		} else {
			std::cout << "模板已加载：" << path << endl;
		}
	} else {
		Mat bgr = raw;
		if (raw.channels() == 1) {
			tpl.gray = raw;
		} else {
			cvtColor(bgr, tpl.gray, COLOR_BGR2GRAY);
		}
		std::cout << "模板已加载：" << path << endl;
	}
	return tpl;
}

static std::string joinPath(const std::string& dir, const std::string& file) {
	if (dir.empty()) {
		return file;
	}
	char last = dir.back();
	if (last == '\\' || last == '/') {
		return dir + file;
	}
	return dir + "\\" + file;
}

static std::string toLowerAscii(std::string s) {
	for (char& c : s) {
		if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
	}
	return s;
}

static bool isFishAltIconFilename(const std::string& file) {
	std::string f = toLowerAscii(file);
	const std::string prefix = "fish_icon_alt";
	const std::string suffix = ".png";
	if (f.size() < prefix.size() + suffix.size()) return false;
	if (f.rfind(prefix, 0) != 0) return false; // must start with prefix
	if (f.compare(f.size() - suffix.size(), suffix.size(), suffix) != 0) return false; // must end with .png
	std::string mid = f.substr(prefix.size(), f.size() - prefix.size() - suffix.size());
	if (mid.empty()) return true; // fish_icon_alt.png
	for (char c : mid) {
		if (c < '0' || c > '9') return false;
	}
	return true; // fish_icon_alt123.png
}

static int parseFishAltIconIndex(const std::string& file) {
	std::string f = toLowerAscii(file);
	const std::string prefix = "fish_icon_alt";
	const std::string suffix = ".png";
	if (!isFishAltIconFilename(file)) return -1;
	std::string mid = f.substr(prefix.size(), f.size() - prefix.size() - suffix.size());
	if (mid.empty()) return -1;
	int v = 0;
	for (char c : mid) {
		if (c < '0' || c > '9') return -1;
		int d = (int)(c - '0');
		if (v > 100000000) return -1; // 防溢出/异常文件名
		v = v * 10 + d;
	}
	return v;
}

static std::vector<std::string> listFilesByWildcard(const std::string& dir, const std::string& wildcard) {
	std::vector<std::string> out;
	std::string query = joinPath(dir, wildcard);
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

	// 排序：fish_icon_alt.png（无数字）最前，其余按数字升序，最后按字典序兜底
	std::sort(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
		bool aOk = isFishAltIconFilename(a);
		bool bOk = isFishAltIconFilename(b);
		if (aOk != bOk) return aOk; // 合法命名优先
		if (!aOk) return toLowerAscii(a) < toLowerAscii(b);
		int ai = parseFishAltIconIndex(a);
		int bi = parseFishAltIconIndex(b);
		if (ai != bi) {
			// -1（无数字）排前面
			if (ai < 0) return true;
			if (bi < 0) return false;
			return ai < bi;
		}
		return toLowerAscii(a) < toLowerAscii(b);
	});
	out.erase(std::unique(out.begin(), out.end(), [](const std::string& a, const std::string& b) {
		return toLowerAscii(a) == toLowerAscii(b);
	}), out.end());

	return out;
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
	loadConfig();

	// 获取窗口句柄
	SetConsoleTitle(L"VRChat FISH! 自动钓鱼 (Draft)");
	std::cout << "    VRChat FISH! 自动钓鱼 (Draft)" << endl << endl;
	std::cout << "  注意：本程序仅学习使用，请遵守相关规则。" << endl << endl;
	std::cout << "  提示：该模式需要使用鼠标左键按住/松开控制，小窗体建议固定为 "
		<< config.target_width << "*" << config.target_height << endl << endl;

	HWND hwnd = findWindowByClassAndTitleContains(config.window_class, config.window_title_contains);
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

	params.tpl_vr_bite_excl_bottom = loadGrayTplFromFile(joinPath(config.resource_dir, config.tpl_bite_exclamation_bottom));
	params.tpl_vr_bite_excl_full = loadGrayTplFromFile(joinPath(config.resource_dir, config.tpl_bite_exclamation_full));
	params.tpl_vr_minigame_bar_full = loadGrayTplFromFile(joinPath(config.resource_dir, config.tpl_minigame_bar_full));
	params.tpl_vr_player_slider = loadGrayTplFromFile(joinPath(config.resource_dir, config.tpl_player_slider));

	// 鱼图标模板：fish_icon + fish_icon_alt*.png（支持 fish_icon_alt3.png / fish_icon_alt10.png ...）
	params.tpl_vr_fish_icons.clear();
	params.tpl_vr_fish_icon_files.clear();
	std::vector<std::string> seenFishFilesLower;
	auto addFishTplFile = [&](const std::string& file, bool required, g_params::GrayTpl* legacyOut) {
		if (file.empty()) return;
		std::string key = toLowerAscii(file);
		if (std::find(seenFishFilesLower.begin(), seenFishFilesLower.end(), key) != seenFishFilesLower.end()) {
			return;
		}
		g_params::GrayTpl tpl = required
			? loadGrayTplFromFile(joinPath(config.resource_dir, file))
			: tryLoadGrayTplFromFile(joinPath(config.resource_dir, file));
		if (tpl.empty()) {
			return; // optional failed
		}
		seenFishFilesLower.push_back(key);
		if (legacyOut) *legacyOut = tpl;
		params.tpl_vr_fish_icons.push_back(tpl);
		params.tpl_vr_fish_icon_files.push_back(file);
	};

	// 主模板：必须存在
	addFishTplFile(config.tpl_fish_icon, true, &params.tpl_vr_fish_icon);
	// 兼容旧配置：可选
	addFishTplFile(config.tpl_fish_icon_alt, false, &params.tpl_vr_fish_icon_alt);
	addFishTplFile(config.tpl_fish_icon_alt2, false, &params.tpl_vr_fish_icon_alt2);
	// 自动扫描目录：fish_icon_alt数字.png / fish_icon_alt.png
	for (const std::string& f : listFilesByWildcard(config.resource_dir, "fish_icon_alt*.png")) {
		if (!isFishAltIconFilename(f)) continue;
		addFishTplFile(f, false, nullptr);
	}

	if (params.tpl_vr_fish_icons.empty()) {
		std::cout << "未加载到任何鱼图标模板，请检查 Resource-VRChat/ 与 tpl_fish_icon 配置。" << endl;
		exit();
	}
}

struct TplMatch {
	Point topLeft{};
	Point center{};
	Rect rect{};
	double score = 0.0;
};

static Rect clampRect(Rect r, const Size& bounds) {
	if (r.x < 0) {
		r.width += r.x;
		r.x = 0;
	}
	if (r.y < 0) {
		r.height += r.y;
		r.y = 0;
	}
	if (r.x + r.width > bounds.width) {
		r.width = bounds.width - r.x;
	}
	if (r.y + r.height > bounds.height) {
		r.height = bounds.height - r.y;
	}
	if (r.width < 0) r.width = 0;
	if (r.height < 0) r.height = 0;
	return r;
}

static Rect centerThirdStripRoi(const Size& bounds) {
	int w = bounds.width;
	int h = bounds.height;
	int x1 = w / 3;
	int x2 = (w * 2) / 3;
	return clampRect(Rect(x1, 0, x2 - x1, h), bounds);
}

TplMatch matchBest(const Mat& srcGray, const g_params::GrayTpl& tpl, int defaultMethod = TM_CCOEFF_NORMED) {
	TplMatch out{};
	const Mat& tplGray = tpl.gray;
	if (srcGray.empty() || tplGray.empty()) {
		return out;
	}
	if (srcGray.cols < tplGray.cols || srcGray.rows < tplGray.rows) {
		return out;
	}
	Mat result;
	int method = defaultMethod;
	if (!tpl.mask.empty()) {
		matchTemplate(srcGray, tplGray, result, method, tpl.mask);
	} else {
		matchTemplate(srcGray, tplGray, result, method);
	}
	double minVal = 0.0, maxVal = 0.0;
	Point minLoc{}, maxLoc{};
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
	Point bestLoc = maxLoc;
	double score = maxVal;
	if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED) {
		bestLoc = minLoc;
		score = 1.0 - minVal;
	}
	if (!std::isfinite(score)) {
		score = 0.0;
	}
	out.topLeft = bestLoc;
	out.rect = Rect(bestLoc.x, bestLoc.y, tplGray.cols, tplGray.rows);
	out.center = Point(bestLoc.x + tplGray.cols / 2, bestLoc.y + tplGray.rows / 2);
	out.score = score;
	return out;
}

TplMatch matchBestRoi(const Mat& srcGray, const g_params::GrayTpl& tpl, Rect roi, int method = TM_CCOEFF_NORMED) {
	TplMatch out{};
	if (srcGray.empty() || tpl.empty()) {
		return out;
	}
	roi = clampRect(roi, srcGray.size());
	if (roi.width < tpl.cols() || roi.height < tpl.rows()) {
		return out;
	}
	Mat sub = srcGray(roi);
	out = matchBest(sub, tpl, method);
	out.topLeft += roi.tl();
	out.center += roi.tl();
	out.rect.x += roi.x;
	out.rect.y += roi.y;
	return out;
}

static TplMatch matchBestRoiMultiScaleByScales(const Mat& srcGray, const g_params::GrayTpl& tpl, Rect roi,
	const double* scales, int scaleCount, int method = TM_CCOEFF_NORMED, double* bestScaleOut = nullptr) {
	TplMatch best{};
	if (srcGray.empty() || tpl.empty()) return best;
	roi = clampRect(roi, srcGray.size());
	if (roi.width <= 0 || roi.height <= 0) return best;
	Mat sub = srcGray(roi);

	double bestScale = 1.0;
	if (!scales || scaleCount <= 0) {
		// fallback: single scale=1.0
		g_params::GrayTpl scaled{};
		scaled.gray = tpl.gray;
		scaled.mask = tpl.mask;
		if (sub.cols >= scaled.gray.cols && sub.rows >= scaled.gray.rows) {
			TplMatch m = matchBest(sub, scaled, method);
			best = m;
			bestScale = 1.0;
		}
	} else {
		for (int i = 0; i < scaleCount; i++) {
			double s = scales[i];
			if (!std::isfinite(s) || s <= 0.0) {
				continue;
			}
			g_params::GrayTpl scaled{};
			if (std::abs(s - 1.0) < 1e-6) {
				scaled.gray = tpl.gray;
				scaled.mask = tpl.mask;
			} else {
				int tw = std::max(1, (int)std::round(tpl.gray.cols * s));
				int th = std::max(1, (int)std::round(tpl.gray.rows * s));
				resize(tpl.gray, scaled.gray, Size(tw, th), 0, 0, INTER_AREA);
				if (!tpl.mask.empty()) {
					resize(tpl.mask, scaled.mask, Size(tw, th), 0, 0, INTER_NEAREST);
				}
			}
			if (sub.cols < scaled.gray.cols || sub.rows < scaled.gray.rows) continue;
			TplMatch m = matchBest(sub, scaled, method);
			if (m.score > best.score) {
				best = m;
				bestScale = s;
			}
		}
	}
	// 坐标从 sub-ROI 空间偏移回原图空间
	best.topLeft += roi.tl();
	best.center += roi.tl();
	best.rect.x += roi.x;
	best.rect.y += roi.y;
	if (bestScaleOut) *bestScaleOut = bestScale;
	return best;
}

static TplMatch matchBestRoiMultiScale(const Mat& srcGray, const g_params::GrayTpl& tpl, Rect roi, int method = TM_CCOEFF_NORMED, double* bestScaleOut = nullptr) {
	const double fishScales[4] = {
		config.fish_scale_1,
		config.fish_scale_2,
		config.fish_scale_3,
		config.fish_scale_4
	};
	return matchBestRoiMultiScaleByScales(srcGray, tpl, roi, fishScales, 4, method, bestScaleOut);
}

// 单尺度模板匹配（用于控制循环的快速路径）
static TplMatch matchBestRoiAtScale(const Mat& srcGray, const g_params::GrayTpl& tpl, Rect roi, double scale, int method = TM_CCOEFF_NORMED) {
	TplMatch out{};
	if (srcGray.empty() || tpl.empty()) return out;
	roi = clampRect(roi, srcGray.size());
	if (roi.width <= 0 || roi.height <= 0) return out;
	Mat sub = srcGray(roi);

	g_params::GrayTpl scaled{};
	if (std::abs(scale - 1.0) < 1e-6) {
		scaled.gray = tpl.gray;
		scaled.mask = tpl.mask;
	} else {
		int tw = std::max(1, (int)std::round(tpl.gray.cols * scale));
		int th = std::max(1, (int)std::round(tpl.gray.rows * scale));
		resize(tpl.gray, scaled.gray, Size(tw, th), 0, 0, INTER_AREA);
		if (!tpl.mask.empty()) {
			resize(tpl.mask, scaled.mask, Size(tw, th), 0, 0, INTER_NEAREST);
		}
	}
	if (sub.cols < scaled.gray.cols || sub.rows < scaled.gray.rows) return out;
	out = matchBest(sub, scaled, method);
	out.topLeft += roi.tl();
	out.center += roi.tl();
	out.rect.x += roi.x;
	out.rect.y += roi.y;
	return out;
}

static void sortUniqueScales(std::vector<double>& scales) {
	std::sort(scales.begin(), scales.end());
	std::vector<double> uniq;
	uniq.reserve(scales.size());
	for (double s : scales) {
		if (!std::isfinite(s) || s <= 0.0) continue;
		if (uniq.empty() || std::abs(uniq.back() - s) > 1e-6) {
			uniq.push_back(s);
		}
	}
	scales.swap(uniq);
}

static void sortUniqueAngles(std::vector<double>& angles) {
	std::sort(angles.begin(), angles.end());
	std::vector<double> uniq;
	uniq.reserve(angles.size());
	for (double a : angles) {
		if (!std::isfinite(a)) continue;
		if (uniq.empty() || std::abs(uniq.back() - a) > 1e-6) {
			uniq.push_back(a);
		}
	}
	angles.swap(uniq);
}

static std::vector<double> buildScaleRange(double minScale, double maxScale, double step, int maxCount = 128) {
	std::vector<double> scales;
	if (!std::isfinite(minScale) || !std::isfinite(maxScale) || !std::isfinite(step)) return scales;
	if (step <= 0.0) return scales;
	if (minScale > maxScale) std::swap(minScale, maxScale);
	if (minScale <= 0.0 || maxScale <= 0.0) return scales;
	if (maxCount < 2) maxCount = 2;

	double span = maxScale - minScale;
	int count = (int)std::floor(span / step + 1e-9) + 1;
	if (count < 1) count = 1;
	if (count > maxCount) {
		// 防止极端配置导致卡死：自动放大步长，让总匹配次数受控
		step = (span <= 0.0) ? step : (span / (double)(maxCount - 1));
		count = maxCount;
	}

	scales.reserve(count + 1);
	for (int i = 0; i < count; i++) {
		double s = minScale + step * (double)i;
		if (!std::isfinite(s) || s <= 0.0) continue;
		scales.push_back(s);
	}
	// 强制包含 maxScale（避免浮点累加误差导致漏掉端点）
	if (!scales.empty()) {
		double last = scales.back();
		if (std::abs(last - maxScale) > step * 0.25) {
			scales.push_back(maxScale);
		}
	}
	sortUniqueScales(scales);
	return scales;
}

static std::vector<double> buildAngleRange(double minAngle, double maxAngle, double step, int maxCount = 256) {
	std::vector<double> angles;
	if (!std::isfinite(minAngle) || !std::isfinite(maxAngle) || !std::isfinite(step)) return angles;
	if (step <= 0.0) return angles;
	if (minAngle > maxAngle) std::swap(minAngle, maxAngle);
	if (maxCount < 2) maxCount = 2;

	double span = maxAngle - minAngle;
	int count = (int)std::floor(span / step + 1e-9) + 1;
	if (count < 1) count = 1;
	if (count > maxCount) {
		step = (span <= 0.0) ? step : (span / (double)(maxCount - 1));
		count = maxCount;
	}

	angles.reserve(count + 1);
	for (int i = 0; i < count; i++) {
		double a = minAngle + step * (double)i;
		if (!std::isfinite(a)) continue;
		angles.push_back(a);
	}
	if (!angles.empty()) {
		double last = angles.back();
		if (std::abs(last - maxAngle) > step * 0.25) {
			angles.push_back(maxAngle);
		}
	}
	sortUniqueAngles(angles);
	return angles;
}

static g_params::GrayTpl makeScaledTpl(const g_params::GrayTpl& tpl, double scale) {
	g_params::GrayTpl scaled{};
	if (tpl.empty()) return scaled;

	if (!std::isfinite(scale) || scale <= 0.0 || std::abs(scale - 1.0) < 1e-6) {
		scaled.gray = tpl.gray;
		scaled.mask = tpl.mask;
		return scaled;
	}

	int tw = std::max(1, (int)std::round(tpl.gray.cols * scale));
	int th = std::max(1, (int)std::round(tpl.gray.rows * scale));
	resize(tpl.gray, scaled.gray, Size(tw, th), 0, 0, INTER_AREA);
	if (!tpl.mask.empty()) {
		resize(tpl.mask, scaled.mask, Size(tw, th), 0, 0, INTER_NEAREST);
	}
	return scaled;
}

static g_params::GrayTpl rotateTplKeepAll(const g_params::GrayTpl& tpl, double angleDeg) {
	g_params::GrayTpl out{};
	if (tpl.empty()) return out;
	if (!std::isfinite(angleDeg) || std::abs(angleDeg) < 1e-6) {
		out.gray = tpl.gray;
		out.mask = tpl.mask;
		return out;
	}

	int w = tpl.gray.cols;
	int h = tpl.gray.rows;
	Point2f center((float)w / 2.0f, (float)h / 2.0f);

	// 计算旋转后完整包围盒，避免裁剪
	RotatedRect rr(center, Size2f((float)w, (float)h), (float)angleDeg);
	Rect2f bbox = rr.boundingRect2f();
	int outW = std::max(1, (int)std::ceil(bbox.width));
	int outH = std::max(1, (int)std::ceil(bbox.height));

	Mat M = getRotationMatrix2D(center, angleDeg, 1.0);
	// 平移：让旋转后的图像位于输出画布中心
	M.at<double>(0, 2) += (double)outW / 2.0 - center.x;
	M.at<double>(1, 2) += (double)outH / 2.0 - center.y;

	warpAffine(tpl.gray, out.gray, M, Size(outW, outH), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
	if (!tpl.mask.empty()) {
		warpAffine(tpl.mask, out.mask, M, Size(outW, outH), INTER_NEAREST, BORDER_CONSTANT, Scalar(0));
	}
	return out;
}

static TplMatch matchBestRoiAtScaleAndAngle(
	const Mat& srcGray,
	const g_params::GrayTpl& tpl,
	Rect roi,
	double scale,
	double angleDeg,
	int method = TM_CCOEFF_NORMED
) {
	if (!std::isfinite(scale) || scale <= 0.0) scale = 1.0;
	if (!std::isfinite(angleDeg)) angleDeg = 0.0;
	g_params::GrayTpl scaled = makeScaledTpl(tpl, scale);
	g_params::GrayTpl rotated = rotateTplKeepAll(scaled, angleDeg);
	return matchBestRoi(srcGray, rotated, roi, method);
}

struct ScaleMatch {
	double scale = 1.0;
	TplMatch match{};
};

static TplMatch matchBestRoiMultiScaleCoarseToFine(
	const Mat& srcGray,
	const g_params::GrayTpl& tpl,
	Rect roi,
	const std::vector<double>& coarseScales,
	int refineTopK,
	double refineRadius,
	double refineStep,
	int method = TM_CCOEFF_NORMED,
	double* bestScaleOut = nullptr
) {
	TplMatch best{};
	if (srcGray.empty() || tpl.empty()) return best;

	// 粗尺度匹配
	double bestScale = 1.0;
	std::vector<ScaleMatch> coarseMatches;
	coarseMatches.reserve(coarseScales.size());

	double coarseMin = 0.0, coarseMax = 0.0;
	bool hasCoarseMinMax = false;

	for (double s : coarseScales) {
		if (!std::isfinite(s) || s <= 0.0) continue;

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

	// 没有有效尺度：fallback 到 1.0
	if (coarseMatches.empty()) {
		best = matchBestRoiAtScale(srcGray, tpl, roi, 1.0, method);
		bestScale = 1.0;
		if (bestScaleOut) *bestScaleOut = bestScale;
		return best;
	}

	// 细分：围绕粗匹配 topK 的尺度再扫一遍
	if (refineTopK < 1) refineTopK = 0;
	if (!std::isfinite(refineRadius) || refineRadius <= 0.0) refineTopK = 0;
	if (!std::isfinite(refineStep) || refineStep <= 0.0) refineTopK = 0;
	if (!hasCoarseMinMax || coarseMax <= 0.0 || coarseMin <= 0.0) refineTopK = 0;

	if (refineTopK > 0) {
		std::sort(coarseMatches.begin(), coarseMatches.end(), [](const ScaleMatch& a, const ScaleMatch& b) {
			return a.match.score > b.match.score;
		});

		std::vector<double> refineScales;
		refineScales.reserve((size_t)refineTopK * 16);

		for (int i = 0; i < (int)coarseMatches.size() && i < refineTopK; i++) {
			const ScaleMatch& cand = coarseMatches[(size_t)i];
			if (cand.match.score <= 0.0) break;

			double s1 = cand.scale - refineRadius;
			double s2 = cand.scale + refineRadius;
			if (s1 < coarseMin) s1 = coarseMin;
			if (s2 > coarseMax) s2 = coarseMax;
			if (s2 <= s1) continue;

			std::vector<double> seg = buildScaleRange(s1, s2, refineStep, 96);
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

	if (bestScaleOut) *bestScaleOut = bestScale;
	return best;
}

struct AngleMatch {
	double angleDeg = 0.0;
	TplMatch match{};
};

static TplMatch matchBestRoiMultiAngleAtScaleCoarseToFine(
	const Mat& srcGray,
	const g_params::GrayTpl& tpl,
	Rect roi,
	double scale,
	const std::vector<double>& coarseAngles,
	int refineTopK,
	double refineRadius,
	double refineStep,
	int method = TM_CCOEFF_NORMED,
	double* bestAngleOut = nullptr
) {
	TplMatch best{};
	if (srcGray.empty() || tpl.empty()) return best;

	g_params::GrayTpl scaledTpl = makeScaledTpl(tpl, scale);
	if (scaledTpl.empty()) return best;

	double bestAngle = 0.0;
	std::vector<AngleMatch> coarseMatches;
	coarseMatches.reserve(coarseAngles.size());

	double coarseMin = 0.0, coarseMax = 0.0;
	bool hasCoarseMinMax = false;

	for (double a : coarseAngles) {
		if (!std::isfinite(a)) continue;
		if (!hasCoarseMinMax) {
			coarseMin = coarseMax = a;
			hasCoarseMinMax = true;
		} else {
			if (a < coarseMin) coarseMin = a;
			if (a > coarseMax) coarseMax = a;
		}

		g_params::GrayTpl rotated = rotateTplKeepAll(scaledTpl, a);
		TplMatch m = matchBestRoi(srcGray, rotated, roi, method);
		coarseMatches.push_back(AngleMatch{ a, m });
		if (m.score > best.score) {
			best = m;
			bestAngle = a;
		}
	}

	if (coarseMatches.empty()) {
		// fallback: angle=0
		g_params::GrayTpl rotated = rotateTplKeepAll(scaledTpl, 0.0);
		best = matchBestRoi(srcGray, rotated, roi, method);
		bestAngle = 0.0;
		if (bestAngleOut) *bestAngleOut = bestAngle;
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
		refineAngles.reserve((size_t)refineTopK * 16);

		for (int i = 0; i < (int)coarseMatches.size() && i < refineTopK; i++) {
			const AngleMatch& cand = coarseMatches[(size_t)i];
			if (cand.match.score <= 0.0) break;

			double a1 = cand.angleDeg - refineRadius;
			double a2 = cand.angleDeg + refineRadius;
			if (a1 < coarseMin) a1 = coarseMin;
			if (a2 > coarseMax) a2 = coarseMax;
			if (a2 <= a1) continue;

			std::vector<double> seg = buildAngleRange(a1, a2, refineStep, 128);
			refineAngles.insert(refineAngles.end(), seg.begin(), seg.end());
		}

		sortUniqueAngles(refineAngles);

		for (double a : refineAngles) {
			g_params::GrayTpl rotated = rotateTplKeepAll(scaledTpl, a);
			TplMatch m = matchBestRoi(srcGray, rotated, roi, method);
			if (m.score > best.score) {
				best = m;
				bestAngle = a;
			}
		}
	}

	if (bestAngleOut) *bestAngleOut = bestAngle;
	return best;
}

static std::vector<double> buildTrackBarScales() {
	std::vector<double> scales;
	// 优先使用 range（0.9~2.0, step=0.1），便于适配不同分辨率/UI缩放
	if (std::isfinite(config.track_scale_min) && std::isfinite(config.track_scale_max)
		&& std::isfinite(config.track_scale_step)
		&& config.track_scale_min > 0.0 && config.track_scale_max > 0.0 && config.track_scale_step > 0.0
		&& config.track_scale_max >= config.track_scale_min) {
		scales = buildScaleRange(config.track_scale_min, config.track_scale_max, config.track_scale_step, 128);
	}
	if (scales.empty()) {
		scales = {
			config.track_scale_1,
			config.track_scale_2,
			config.track_scale_3,
			config.track_scale_4
		};
		sortUniqueScales(scales);
	}
	return scales;
}

static TplMatch matchBestRoiTrackBarAutoScale(
	const Mat& srcGray,
	const g_params::GrayTpl& tpl,
	Rect roi,
	int method = TM_CCOEFF_NORMED,
	double* bestScaleOut = nullptr,
	double* bestAngleOut = nullptr
) {
	std::vector<double> coarse = buildTrackBarScales();
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
		&bestScale
	);
	double bestAngle = 0.0;
	if (std::isfinite(config.track_angle_min) && std::isfinite(config.track_angle_max)
		&& std::isfinite(config.track_angle_step)
		&& config.track_angle_step > 0.0) {
		double aMin = config.track_angle_min;
		double aMax = config.track_angle_max;
		if (aMin > aMax) std::swap(aMin, aMax);
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
				&bestAngle
			);
		}
	}
	if (bestScaleOut) *bestScaleOut = bestScale;
	if (bestAngleOut) *bestAngleOut = bestAngle;
	return best;
}

static void activateGameWindowInner(bool forceCursorCenter) {
	HWND hwnd = params.hwnd;
	if (!hwnd) {
		return;
	}
	ShowWindow(hwnd, SW_RESTORE);
	SetForegroundWindow(hwnd);
	BringWindowToTop(hwnd);
	if (config.vr_debug) {
		HWND fg = GetForegroundWindow();
		if (fg != hwnd) {
			static unsigned long long lastWarnMs = 0;
			unsigned long long t = GetTickCount64();
			if (t - lastWarnMs > 2000) {
				std::cout << "[vrchat_fish] warn: foreground hwnd mismatch (fg=" << fg << " vrchat=" << hwnd << ")" << endl;
				lastWarnMs = t;
			}
		}
	}

	// SendInput 会在"当前鼠标指针所在窗口"分发消息；如果鼠标在窗口外会导致点击无效
	RECT r = params.rect;
	if (r.right > r.left && r.bottom > r.top) {
		POINT p{};
		if (GetCursorPos(&p)) {
			if (forceCursorCenter || p.x < r.left || p.x >= r.right || p.y < r.top || p.y >= r.bottom) {
				SetCursorPos((r.left + r.right) / 2, (r.top + r.bottom) / 2);
			}
		}
	}
}

static void activateGameWindow() {
	activateGameWindowInner(false);
}

static void sendMouseLeftRaw(DWORD flags, const char* phaseTag) {
	INPUT input{};
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = flags;
	input.mi.time = NULL;
	if (SendInput(1, &input, sizeof(INPUT)) != 1 && config.vr_debug) {
		std::cout << "[vrchat_fish] SendInput " << phaseTag << " failed err=" << GetLastError() << endl;
	}
}

static void mouseLeftDown() {
	activateGameWindow();
	sendMouseLeftRaw(MOUSEEVENTF_LEFTDOWN, "leftdown");
}

static void mouseLeftUp() {
	activateGameWindow();
	sendMouseLeftRaw(MOUSEEVENTF_LEFTUP, "leftup");
}

static void mouseLeftClickCentered(int delayMs = 40) {
	activateGameWindowInner(true);
	sendMouseLeftRaw(MOUSEEVENTF_LEFTDOWN, "leftdown");
	Sleep(delayMs);
	sendMouseLeftRaw(MOUSEEVENTF_LEFTUP, "leftup");
}

static void mouseMoveRelative(int dx, int dy, const char* phaseTag) {
	if (dx == 0 && dy == 0) {
		return;
	}
	activateGameWindow();
	INPUT input{};
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_MOVE;
	input.mi.dx = dx;
	input.mi.dy = dy;
	input.mi.time = NULL;
	if (SendInput(1, &input, sizeof(INPUT)) != 1 && config.vr_debug) {
		std::cout << "[vrchat_fish] SendInput " << phaseTag << " failed err=" << GetLastError() << endl;
	}
}

static void keyTapVk(WORD vk, int delayMs = 30) {
	activateGameWindow();
	INPUT input{};
	input.type = INPUT_KEYBOARD;
	input.ki.wVk = vk;
	input.ki.wScan = (WORD)MapVirtualKey(vk, MAPVK_VK_TO_VSC);
	input.ki.dwFlags = 0;
	if (SendInput(1, &input, sizeof(INPUT)) != 1 && config.vr_debug) {
		std::cout << "[vrchat_fish] SendInput keydown failed err=" << GetLastError() << endl;
	}
	Sleep(delayMs);
	input.ki.dwFlags = KEYEVENTF_KEYUP;
	if (SendInput(1, &input, sizeof(INPUT)) != 1 && config.vr_debug) {
		std::cout << "[vrchat_fish] SendInput keyup failed err=" << GetLastError() << endl;
	}
}

static bool ensureDirExists(const std::string& dir) {
	if (dir.empty()) {
		return false;
	}

	std::string path = dir;
	std::replace(path.begin(), path.end(), '/', '\\');

	DWORD attrs = GetFileAttributesA(path.c_str());
	if (attrs != INVALID_FILE_ATTRIBUTES) {
		return (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0;
	}

	int rc = SHCreateDirectoryExA(NULL, path.c_str(), NULL);
	if (rc == ERROR_SUCCESS || rc == ERROR_ALREADY_EXISTS || rc == ERROR_FILE_EXISTS) {
		return true;
	}

	// Re-check in case another process created it.
	attrs = GetFileAttributesA(path.c_str());
	return (attrs != INVALID_FILE_ATTRIBUTES) && ((attrs & FILE_ATTRIBUTE_DIRECTORY) != 0);
}

static std::string makeDebugPath(const std::string& tag) {
	std::string dir = config.vr_debug_dir.empty() ? "debug_vrchat" : config.vr_debug_dir;
	ensureDirExists(dir);
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

enum class VrFishState {
	Cast,
	WaitBite,
	EnterMinigame,
	ControlMinigame,
	PostMinigame,
};

static unsigned long long nowMs() {
	return GetTickCount64();
}

static bool detectBite(const Mat& gray, TplMatch* matchOut = nullptr) {
	TplMatch m = matchBest(gray, params.tpl_vr_bite_excl_bottom);
	if (m.score < config.bite_threshold) {
		TplMatch m2 = matchBest(gray, params.tpl_vr_bite_excl_full);
		if (m2.score > m.score) {
			m = m2;
		}
	}
	if (matchOut) *matchOut = m;
	return m.score >= config.bite_threshold;
}

// ── 颜色检测：在竖条上找到玩家滑块（亮色区域）的上下边界 ──
// 在 barX 附近的窄列中，用亮度阈值找最长连续亮色段作为滑块
// 返回 sliderTop, sliderBottom（全图坐标），成功返回 true
static bool detectSliderBounds(const Mat& gray, int barX, const Rect& searchRoi,
	int* sliderTopOut, int* sliderBottomOut, int* sliderCenterYOut,
	int brightnessThresh = 180, int minSliderHeight = 15) {

	// 取 barX 附近 ±halfW 像素宽的竖条取中值，抗噪声
	int halfW = config.slider_detect_half_width;
	if (halfW < 0) halfW = 0;
	int x1 = max(searchRoi.x, barX - halfW);
	int x2 = min(searchRoi.x + searchRoi.width, barX + halfW + 1);
	if (x2 <= x1) return false;

	int y1 = searchRoi.y;
	int y2 = searchRoi.y + searchRoi.height;
	if (y1 < 0) y1 = 0;
	if (y2 > gray.rows) y2 = gray.rows;
	if (y2 <= y1) return false;

	// 对每行取该窄条的平均亮度，收集所有亮色段
	struct BrightRun { int start; int len; };
	std::vector<BrightRun> runs;
	int curRunStart = -1, curRunLen = 0;

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
	// 收尾
	if (curRunLen > 0) {
		runs.push_back({ curRunStart, curRunLen });
	}

	if (runs.empty()) return false;

	// 合并相邻亮色段：如果两段间隔 <= maxGap 像素，视为同一滑块（鱼图标造成的分割）
	int maxGap = config.slider_detect_merge_gap;
	if (maxGap < 0) maxGap = 0;
	std::vector<BrightRun> merged;
	merged.push_back(runs[0]);
	for (size_t i = 1; i < runs.size(); i++) {
		BrightRun& last = merged.back();
		int lastEnd = last.start + last.len;
		int gap = runs[i].start - lastEnd;
		if (gap <= maxGap) {
			// 合并：扩展到包含新段
			last.len = (runs[i].start + runs[i].len) - last.start;
		} else {
			merged.push_back(runs[i]);
		}
	}

	// 从合并后的段中找最长的
	int bestIdx = 0;
	for (size_t i = 1; i < merged.size(); i++) {
		if (merged[i].len > merged[bestIdx].len) bestIdx = (int)i;
	}
	int bestRunStart = merged[bestIdx].start;
	int bestRunLen = merged[bestIdx].len;

	if (bestRunLen < minSliderHeight) return false;
	// 防止误把整条轨道（或背景高亮）当成滑块
	if (bestRunLen >= (int)((y2 - y1) * 0.95)) return false;

	if (sliderTopOut) *sliderTopOut = bestRunStart;
	if (sliderBottomOut) *sliderBottomOut = bestRunStart + bestRunLen;
	if (sliderCenterYOut) *sliderCenterYOut = bestRunStart + bestRunLen / 2;
	return true;
}

// ── 颜色检测（整段扫）：在整个轨道 ROI 内找滑块亮段的上下边界 ──
// 思路：对 ROI 的每一行统计“亮色像素数量”，连续超过阈值的区间视为滑块
// 这样就不依赖滑块模板给出的 barX，也更不怕被鱼遮挡导致模板偏移
static bool detectSliderBoundsWide(const Mat& gray, const Rect& searchRoi,
	int* sliderTopOut, int* sliderBottomOut, int* sliderCenterYOut,
	int brightnessThresh = 180, int minSliderHeight = 15) {

	Rect roi = clampRect(searchRoi, gray.size());
	if (roi.width <= 0 || roi.height <= 0) return false;

	int x1 = roi.x;
	int x2 = roi.x + roi.width;
	int y1 = roi.y;
	int y2 = roi.y + roi.height;
	if (x2 <= x1 || y2 <= y1) return false;

	// 每行至少需要多少个“足够亮”的像素才算滑块范围
	// ROI 宽度通常在 40~60 左右，滑块宽度约占其中一半；这里取一个较保守的比例
	int minRowBright = roi.width / 8; // 约 12.5%
	if (minRowBright < 4) minRowBright = 4;
	if (minRowBright > roi.width) minRowBright = roi.width;

	struct BrightRun { int start; int len; };
	std::vector<BrightRun> runs;
	int curRunStart = -1, curRunLen = 0;

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

	// 合并相邻亮段（鱼图标挡住滑块会把亮段切开）
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

	// 取最长段
	int bestIdx = 0;
	for (size_t i = 1; i < merged.size(); i++) {
		if (merged[i].len > merged[bestIdx].len) bestIdx = (int)i;
	}
	int bestRunStart = merged[bestIdx].start;
	int bestRunLen = merged[bestIdx].len;

	if (bestRunLen < minSliderHeight) return false;
	// 防止误把整条轨道（或背景高亮）当成滑块
	if (bestRunLen >= (int)(roi.height * 0.95)) return false;

	if (sliderTopOut) *sliderTopOut = bestRunStart;
	if (sliderBottomOut) *sliderBottomOut = bestRunStart + bestRunLen;
	if (sliderCenterYOut) *sliderCenterYOut = bestRunStart + bestRunLen / 2;
	return true;
}

struct FishSliderResult {
	int fishX, fishY;
	int sliderCenterX, sliderCenterY;
	int sliderTop, sliderBottom;    // 滑块上下边界（颜色检测）
	int sliderHeight;               // 滑块实际高度（像素）
	double fishScore, sliderScore;
	bool hasBounds;                 // 是否成功检测到滑块边界
};

static Point2f affineTransformPoint(const Mat& M, const Point2f& p) {
	// M: 2x3
	Point2f out{};
	if (M.empty() || M.rows != 2 || M.cols != 3) {
		return p;
	}
	out.x = (float)(M.at<double>(0, 0) * p.x + M.at<double>(0, 1) * p.y + M.at<double>(0, 2));
	out.y = (float)(M.at<double>(1, 0) * p.x + M.at<double>(1, 1) * p.y + M.at<double>(1, 2));
	return out;
}

static bool fillFishSliderResult(const Mat& gray, const Rect& roi, const TplMatch& fish, const TplMatch& slider,
	double trackAngleDeg, int fishTplHeightHint, FishSliderResult* result) {
	if (!result) return false;
	*result = FishSliderResult{};

	result->fishScore = fish.score;
	result->sliderScore = slider.score;
	if (fish.score < config.fish_icon_threshold) {
		return false;
	}

	result->fishX = fish.center.x;
	result->fishY = fish.center.y;
	// 不再依赖滑块模板位置：默认用 fishX 作为滑块X（只用于日志/调试，控制只用 Y）
	result->sliderCenterX = fish.center.x;
	result->sliderCenterY = slider.center.y;

	// 颜色检测：优先在整个轨道 ROI 内扫亮度线条（不依赖滑块模板分数/位置）
	// minSliderHeight 过小会把“鱼图标”这种短亮段误当成滑块，这里做个下限保护
	int effectiveMinH = config.slider_min_height;
	int fishTplH = fishTplHeightHint;
	if (fishTplH <= 0) fishTplH = params.tpl_vr_fish_icon.rows();
	if (fishTplH > 0 && effectiveMinH < fishTplH + 5) effectiveMinH = fishTplH + 5;
	int sliderTop = 0, sliderBottom = 0, sliderCenterFromColor = 0;
	{
		Rect r = clampRect(roi, gray.size());
		Mat roiGray = gray(r);
		Rect localRoi(0, 0, roiGray.cols, roiGray.rows);

		// 旋转矫正：轨道检测得到 angle 后，亮度检测也按同角度把 ROI 旋回到“竖直”再扫（提高鲁棒性）
		double a = trackAngleDeg;
		if (!std::isfinite(a)) a = 0.0;
		Mat scanGray = roiGray; // default: no rotation
		Mat rotated;
		Mat M, Minv;
		int barXLocal = fish.center.x - r.x;
		int barYLocal = fish.center.y - r.y;
		if (barXLocal < 0) barXLocal = 0;
		if (barXLocal >= roiGray.cols) barXLocal = roiGray.cols - 1;
		if (barYLocal < 0) barYLocal = 0;
		if (barYLocal >= roiGray.rows) barYLocal = roiGray.rows - 1;

		float barXForMap = (float)barXLocal; // original-local x for mapping
		if (std::abs(a) > 1e-6 && roiGray.cols > 1 && roiGray.rows > 1) {
			Point2f c((float)roiGray.cols / 2.0f, (float)roiGray.rows / 2.0f);
			M = getRotationMatrix2D(c, -a, 1.0);
			Minv = getRotationMatrix2D(c, +a, 1.0);
			warpAffine(roiGray, rotated, M, roiGray.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
			scanGray = rotated;

			// barX 也需要变换到旋转后的坐标系下（用于窄条扫描兜底 + 反变换坐标）
			Point2f barRot = affineTransformPoint(M, Point2f((float)barXLocal, (float)barYLocal));
			barXForMap = barRot.x;
			if (barXForMap < 0.0f) barXForMap = 0.0f;
			if (barXForMap > (float)(roiGray.cols - 1)) barXForMap = (float)(roiGray.cols - 1);
			barXLocal = (int)std::round(barXForMap);
		}

		int t = 0, b = 0, cy = 0;
		bool okColor = detectSliderBoundsWide(scanGray, localRoi,
			&t, &b, &cy,
			config.slider_bright_thresh, effectiveMinH)
			|| detectSliderBounds(scanGray, barXLocal, localRoi,
				&t, &b, &cy,
				config.slider_bright_thresh, effectiveMinH);

		if (okColor) {
			// 坐标反变换回原图坐标（保持与 fishY / fixedTrackRoi.y 一致的全图坐标系）
			if (!Minv.empty()) {
				Point2f pTop = affineTransformPoint(Minv, Point2f(barXForMap, (float)t));
				Point2f pBot = affineTransformPoint(Minv, Point2f(barXForMap, (float)b));
				Point2f pCy = affineTransformPoint(Minv, Point2f(barXForMap, (float)cy));
				sliderTop = (int)std::round(pTop.y) + r.y;
				sliderBottom = (int)std::round(pBot.y) + r.y;
				sliderCenterFromColor = (int)std::round(pCy.y) + r.y;
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
			result->sliderCenterY = sliderCenterFromColor;  // 用颜色检测的中心覆盖模板匹配的中心
			result->hasBounds = true;
		}
	}

	if (!result->hasBounds) {
		// 颜色检测失败 → 退回滑块模板（作为兜底）
		if (slider.score < config.slider_threshold) {
			return false;
		}
		int tplH = params.tpl_vr_player_slider.rows();
		result->sliderTop = slider.center.y - tplH / 2;
		result->sliderBottom = slider.center.y + tplH / 2;
		result->sliderHeight = tplH;
		result->hasBounds = false;
	}

	return true;
}

	// 快速检测（控制循环用）：固定 trackScale/angle + 单鱼模板，仅 2 次 matchTemplate
	// cachedFishTplIdx: params.tpl_vr_fish_icons 的索引（fish_icon + fish_icon_alt*.png）
	static bool detectFishAndSliderFast(const Mat& gray, const Rect& barRect, FishSliderResult* result,
		double trackScale, double trackAngleDeg, int cachedFishTplIdx) {
		Rect roi = clampRect(barRect, gray.size());
		if (roi.width <= 0 || roi.height <= 0) return false;

		double s = trackScale;
		if (!std::isfinite(s) || s <= 0.0) s = 1.0;

		const g_params::GrayTpl* fishTplPtr = &params.tpl_vr_fish_icon;
		if (!params.tpl_vr_fish_icons.empty()) {
			int idx = cachedFishTplIdx;
			if (idx < 0) idx = 0;
			if (idx >= (int)params.tpl_vr_fish_icons.size()) idx = 0;
			fishTplPtr = &params.tpl_vr_fish_icons[(size_t)idx];
		}
		const auto& fishTpl = *fishTplPtr;
		TplMatch fish = matchBestRoiAtScaleAndAngle(gray, fishTpl, roi, s, trackAngleDeg, TM_CCOEFF_NORMED);
		TplMatch slider = matchBestRoi(gray, params.tpl_vr_player_slider, roi, TM_CCORR_NORMED);
		int fishTplHScaled = (int)std::round((double)fishTpl.rows() * s);
		return fillFishSliderResult(gray, roi, fish, slider, trackAngleDeg, fishTplHScaled, result);
	}

	// 全检测（首帧 + 周期性刷新用）：固定 trackScale/angle 下尝试多鱼模板，输出最佳模板索引
	static bool detectFishAndSliderFull(const Mat& gray, const Rect& barRect, FishSliderResult* result,
		double trackScale, double trackAngleDeg, int* bestTplIdxOut) {
		Rect roi = clampRect(barRect, gray.size());
		if (roi.width <= 0 || roi.height <= 0) return false;

		double s = trackScale;
		if (!std::isfinite(s) || s <= 0.0) s = 1.0;

		if (params.tpl_vr_fish_icons.empty()) {
			return false;
		}
		TplMatch fish{};
		int bestIdx = 0;
		for (size_t i = 0; i < params.tpl_vr_fish_icons.size(); i++) {
			const auto& tpl = params.tpl_vr_fish_icons[i];
			TplMatch m = matchBestRoiAtScaleAndAngle(gray, tpl, roi, s, trackAngleDeg, TM_CCOEFF_NORMED);
			if (m.score > fish.score) {
				fish = m;
				bestIdx = (int)i;
			}
		}

		TplMatch slider = matchBestRoi(gray, params.tpl_vr_player_slider, roi, TM_CCORR_NORMED);
		int fishTplHScaled = (int)std::round((double)params.tpl_vr_fish_icons[(size_t)bestIdx].rows() * s);
		bool ok = fillFishSliderResult(gray, roi, fish, slider, trackAngleDeg, fishTplHScaled, result);
		if (!ok) {
			return false;
		}
		if (bestTplIdxOut) *bestTplIdxOut = bestIdx;
		return true;
	}

// ── MLP 权重加载（为后续推理模式预留） ─────────────────────────
struct MlpLayer {
	int in_dim, out_dim;
	std::vector<double> weights; // out_dim × in_dim (row-major)
	std::vector<double> bias;    // out_dim
};

struct MlpModel {
	std::vector<MlpLayer> layers;
	std::vector<double> norm_mean;  // 输入特征归一化均值
	std::vector<double> norm_std;   // 输入特征归一化标准差
	bool loaded = false;
};

static MlpModel g_mlpModel;

static bool loadMlpWeights(const std::string& path, MlpModel& model) {
	std::ifstream f(path);
	if (!f.is_open()) {
		std::cerr << "[ML] 无法打开权重文件: " << path << endl;
		return false;
	}
	model.layers.clear();
	std::string line;
	while (std::getline(f, line)) {
		// 跳过空行和注释
		if (line.empty() || line[0] == '#') continue;
		// 读取 in_dim out_dim
		std::istringstream hdr(line);
		int in_dim, out_dim;
		if (!(hdr >> in_dim >> out_dim)) continue;
		MlpLayer layer;
		layer.in_dim = in_dim;
		layer.out_dim = out_dim;
		layer.weights.resize(out_dim * in_dim);
		layer.bias.resize(out_dim);
		// 读取 weight matrix (out_dim 行, 每行 in_dim 个值)
		for (int r = 0; r < out_dim; r++) {
			if (!std::getline(f, line)) { std::cerr << "[ML] 权重文件格式错误 (weights)" << endl; return false; }
			std::istringstream row(line);
			for (int c = 0; c < in_dim; c++) {
				if (!(row >> layer.weights[r * in_dim + c])) {
					std::cerr << "[ML] 权重文件格式错误 (weight value)" << endl;
					return false;
				}
			}
		}
		// 读取 bias vector (1 行, out_dim 个值)
		if (!std::getline(f, line)) { std::cerr << "[ML] 权重文件格式错误 (bias)" << endl; return false; }
		std::istringstream brow(line);
		for (int r = 0; r < out_dim; r++) {
			if (!(brow >> layer.bias[r])) {
				std::cerr << "[ML] 权重文件格式错误 (bias value)" << endl;
				return false;
			}
		}
		model.layers.push_back(std::move(layer));
	}
	if (model.layers.empty()) {
		std::cerr << "[ML] 权重文件为空" << endl;
		return false;
	}
	model.loaded = true;
	std::cout << "[ML] 已加载模型: " << model.layers.size() << " 层" << endl;

	// 加载归一化参数 (weights_file 路径将 .txt 替换为 _norm.txt)
	std::string normPath = path;
	size_t dotPos = normPath.rfind('.');
	if (dotPos != std::string::npos)
		normPath = normPath.substr(0, dotPos) + "_norm.txt";
	else
		normPath += "_norm";
	std::ifstream nf(normPath);
	if (nf.is_open()) {
		std::string nline;
		model.norm_mean.clear();
		model.norm_std.clear();
		while (std::getline(nf, nline)) {
			if (nline.empty() || nline[0] == '#') continue;
			std::istringstream ns(nline);
			double m, s;
			if (ns >> m >> s) {
				model.norm_mean.push_back(m);
				model.norm_std.push_back(s);
			}
		}
		std::cout << "[ML] 已加载归一化参数: " << model.norm_mean.size() << " 个特征" << endl;
	} else {
		std::cout << "[ML] 警告: 未找到归一化文件 " << normPath << ", 将使用原始输入" << endl;
	}
	return true;
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
			int    cachedFishTplIdx = 0;     // params.tpl_vr_fish_icons 的索引
			bool   hasCachedFishTpl = false;

		// ML 录制模式状态
		std::ofstream recordFile;
		int recordFrame = 0;
		std::deque<int> pressWindow;   // 滑动窗口：最近 N 帧 mousePressed (0/1)
		static const int PRESS_WINDOW_SIZE = 10;

		// VRChat 日志输出文件（用于拟合物理参数，避免手动复制控制台输出）
		std::ofstream vrLogFile;

		// ML 推理模式：加载权重
		if (config.ml_mode == 2 && !g_mlpModel.loaded) {
			if (!loadMlpWeights(config.ml_weights_file, g_mlpModel)) {
				std::cerr << "[ML] 无法加载权重，退回 PD 模式" << endl;
				config.ml_mode = 0;
			}
		}

		auto writeVrLogLine = [&](const std::string& line, bool alsoStdout = true) {
			if (alsoStdout) {
				std::cout << line << endl;
			}
			if (vrLogFile.is_open()) {
				vrLogFile << line << '\n';
			}
		};

			if (!config.vr_log_file.empty()) {
				// 确保目录存在（支持多级目录；路径不含目录则跳过）
				auto dirNameOf = [&](const std::string& path) -> std::string {
					size_t pos = path.find_last_of("/\\");
					if (pos == std::string::npos) return "";
					if (pos == 0) return "";
					return path.substr(0, pos);
				};
				std::string dir = dirNameOf(config.vr_log_file);
				if (!dir.empty()) {
					ensureDirExists(dir);
				}
				vrLogFile.open(config.vr_log_file, std::ios::out | std::ios::app);
				if (!vrLogFile.is_open()) {
					std::cout << "[vrchat_fish] WARN: failed to open vr_log_file=" << config.vr_log_file
						<< " (check working dir / file lock)" << endl;
				}
				else {
					writeVrLogLine("[vrchat_fish] log start file=" + config.vr_log_file, config.vr_debug);
				}
			}

		auto switchState = [&](VrFishState next) {
			if (config.vr_debug || vrLogFile.is_open()) {
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
			if (config.vr_debug || vrLogFile.is_open()) {
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
				if (config.vr_debug || vrLogFile.is_open()) {
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
				if (config.vr_debug || vrLogFile.is_open()) {
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
			bool ok = detectBite(gray, &m);
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

				Rect searchRoi = centerThirdStripRoi(gray.size());
				// ── 首帧：用 minigame_bar_full 模板定位完整轨道，取右半部分作为滑块/鱼检测 ROI ──
				if (!hasFixedTrack) {
					double barScale = 1.0;
					double barAngle = 0.0;
					TplMatch barMatch = matchBestRoiTrackBarAutoScale(
						gray,
						params.tpl_vr_minigame_bar_full,
						searchRoi,
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
						fixedTrackRoi = clampRect(fixedTrackRoi, gray.size());
						hasFixedTrack = true;
						cachedTrackScale = barScale;
						cachedTrackAngle = barAngle;
						// debug: 蓝框=搜索区域, 绿框=模板匹配位置, 红框=最终锁定ROI
						saveDebugFrame(frame, "track_lock", searchRoi, barMatch.rect, fixedTrackRoi);
						if (config.vr_debug || vrLogFile.is_open()) {
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
						if (config.vr_debug || vrLogFile.is_open()) {
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
					ok = detectFishAndSliderFull(gray, matchRoi, &det, cachedTrackScale, cachedTrackAngle, &bestIdx);
					if (ok) {
						cachedFishTplIdx = bestIdx;
						hasCachedFishTpl = true;
					}
					didFullDetect = true;
				} else {
					ok = detectFishAndSliderFast(gray, matchRoi, &det, cachedTrackScale, cachedTrackAngle, cachedFishTplIdx);
					if (!ok) {
						int bestIdx = 0;
						ok = detectFishAndSliderFull(gray, matchRoi, &det, cachedTrackScale, cachedTrackAngle, &bestIdx);
						if (ok) {
							cachedFishTplIdx = bestIdx;
						}
						didFullDetect = true;
					}
				}
	
				unsigned long long detectMs = nowMs() - loopStart;

			if (!ok) {
				if (config.vr_debug || vrLogFile.is_open()) {
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
					if (config.vr_debug || vrLogFile.is_open()) {
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
					// ══ MPC 控制器：轨迹模拟 + 选择最优动作 ══
					//
					// 物理模型（每步）:
					//   velocity = velocity * drag + (pressing ? thrust : gravity)
					//   position += velocity
					//
					// 每帧模拟两条轨迹（按住 vs 松开），选择使鱼
					// 留在滑块安全区内时间更长的那个动作

					double gravity = config.bb_gravity;     // 正值，向下
					double thrust  = config.bb_thrust;      // 负值，向上
					double drag    = config.bb_drag;
					if (drag < 0.5) drag = 0.5;
					if (drag > 1.0) drag = 1.0;
					int horizon = config.bb_sim_horizon;
					if (horizon < 1) horizon = 1;
					if (horizon > 30) horizon = 30;

					double marginRatio = config.bb_margin_ratio;
					if (marginRatio < 0.0) marginRatio = 0.0;
					if (marginRatio > 0.45) marginRatio = 0.45;
					double margin = sliderH * marginRatio;

					// 模拟一条轨迹，返回累积代价（越小越好）
					// 代价 = sum of (鱼到安全区中心的距离 if 在区外, 0 if 在区内)
					double fishVelDecay = config.fish_vel_decay;
					if (fishVelDecay < 0.5) fishVelDecay = 0.5;
					if (fishVelDecay > 1.0) fishVelDecay = 1.0;
					bool fishBounce = (config.fish_bounce_predict != 0);

					auto simCost = [&](bool press) -> double {
						double sVel = smoothVelocity;   // 滑块当前速度
						double sCY  = (double)sliderCY; // 滑块中心 Y
						double fY   = (double)fishY;    // 鱼 Y
						double fVel = smoothFishVel;    // 鱼速度
						double fAcc = smoothFishAccel;  // 鱼加速度
						double cost = 0.0;

						// 轨道物理边界（固定ROI上下各扩展 padY 用于检测，这里扣回）
						// 滑块中心碰到物理边界即触发反弹
						double trackPadY = (double)std::max(0, config.track_pad_y);
						double trackCYMin = (double)fixedTrackRoi.y + trackPadY;
						double trackCYMax = (double)(fixedTrackRoi.y + fixedTrackRoi.height) - trackPadY;
						if (trackCYMin > trackCYMax) trackCYMin = trackCYMax = (trackCYMin + trackCYMax) / 2.0;

						for (int step = 0; step < horizon; step++) {
							// 滑块物理更新
							double accel = press ? thrust : gravity;
							sVel = sVel * drag + accel;
							sCY += sVel;

							// 轨道边界反弹：碰壁后速度反向
							if (sCY < trackCYMin) {
								sCY = trackCYMin;
								sVel = -sVel * 0.8; // 反弹，损失部分动能
							} else if (sCY > trackCYMax) {
								sCY = trackCYMax;
								sVel = -sVel * 0.8;
							}

							// ── 方向A：鱼位置更新（加速度预测 + 速度衰减 + 边界反弹） ──
							fVel += fAcc;                  // 加速度影响速度
							fVel *= fishVelDecay;          // 远期速度逐步衰减（不确定性增大）
							fY += fVel;

							// 鱼碰轨道边界反弹
							if (fishBounce) {
								if (fY < trackCYMin) {
									fY = trackCYMin;
									fVel = -fVel * 0.5;
									fAcc = 0.0; // 反弹后加速度失效
								} else if (fY > trackCYMax) {
									fY = trackCYMax;
									fVel = -fVel * 0.5;
									fAcc = 0.0;
								}
							}

							// 滑块边界
							double halfH = (double)sliderH / 2.0;
							double sTop = sCY - halfH + margin;
							double sBot = sCY + halfH - margin;

							// 代价：鱼偏离安全区的程度
							if (sTop >= sBot) {
								// 滑块太短，按中心距离算代价
								cost += abs(fY - sCY);
							} else if (fY < sTop) {
								cost += (sTop - fY);
							} else if (fY > sBot) {
								cost += (fY - sBot);
							}
							// 在安全区内 → cost += 0
						}

						// 边界减速惩罚：滑块以较高速度接近轨道边界时施加额外代价
						// 目的：让 MPC 主动选择提前制动，避免高速碰壁后剧烈反弹
						double bZone   = config.bb_boundary_zone;
						double bWeight = config.bb_boundary_weight;
						if (bZone > 0.0 && bWeight > 0.0) {
							double distTop = sCY - trackCYMin;
							double distBot = trackCYMax - sCY;
							double distMin = distTop < distBot ? distTop : distBot;
							if (distMin < bZone) {
								double penetration = (bZone - distMin) / bZone; // 0~1，越靠近惩罚越大
								cost += abs(sVel) * penetration * bWeight;
							}
						}
						return cost;
					};

					double costPress   = simCost(true);
					double costRelease = simCost(false);

					bool wantHold;
					if (abs(costPress - costRelease) < 0.5) {
						// 两个动作代价差不多 → 维持当前状态（减少振荡）
						wantHold = holding;
					} else {
						wantHold = (costPress < costRelease);
					}

					// ── 方向D：偏差快速响应 ──
					// 当鱼与滑块偏差大且在快速增长时，直接覆盖MPC决策
					bool reactiveTriggered = false;
					if (config.reactive_override) {
						double deviation = (double)(fishY - sliderCY); // 正=鱼在下方
						double absDev = abs(deviation);
						double devThreshold = (double)sliderH * config.reactive_dev_ratio;
						if (devThreshold < 10.0) devThreshold = 10.0;

						if (absDev > devThreshold && hasPrevDeviation) {
							// 偏差增长速度（归一化到基准帧）
							double devGrowth = (absDev - abs(prevDeviation)) / lastDtRatio;
							if (devGrowth > config.reactive_grow_threshold) {
								// 鱼在下方且偏差在增大 → 应该松开（让滑块往下）
								// 鱼在上方且偏差在增大 → 应该按住（让滑块往上）
								bool reactiveHold = (deviation < 0); // 鱼在上方
								if (reactiveHold != wantHold) {
									wantHold = reactiveHold;
									reactiveTriggered = true;
								}
							}
						}
						prevDeviation = deviation;
						hasPrevDeviation = true;
					}

					if (wantHold && !holding) {
						mouseLeftDown();
						holding = true;
					} else if (!wantHold && holding) {
						mouseLeftUp();
						holding = false;
					}

					if (config.vr_debug || vrLogFile.is_open()) {
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
