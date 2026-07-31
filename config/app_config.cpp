#include "app_config.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "../ini.h"

namespace {

bool isNumeric(const std::string& str) {
	for (char ch : str) {
		if (ch < '0' || ch > '9') {
			return false;
		}
	}
	return !str.empty();
}

bool parseKeyVal(const std::string& s, int* out, std::string* error) {
	if (out == nullptr) {
		if (error != nullptr) {
			*error = "cleanup_reel_key output pointer is null";
		}
		return false;
	}

	if (s.size() > 2 && s.rfind("0x", 0) == 0) {
		try {
			*out = std::stoi(s, nullptr, 16);
			return true;
		} catch (...) {
			if (error != nullptr) {
				std::ostringstream oss;
				oss << "invalid hexadecimal cleanup_reel_key: " << s;
				*error = oss.str();
			}
			return false;
		}
	}
	if (s.length() == 1) {
		if (isNumeric(s)) {
			*out = 0x30 + std::atoi(s.c_str());
			return true;
		}
		*out = s[0];
		return true;
	}
	if (s == "leftClick") {
		*out = 0x01;
		return true;
	}
	if (s == "rightClick") {
		*out = 0x02;
		return true;
	}
	if (s == "ctrl") {
		*out = 17;
		return true;
	}
	if (s == "space") {
		*out = 32;
		return true;
	}
	if (s == "tab") {
		*out = 9;
		return true;
	}
	if (error != nullptr) {
		std::ostringstream oss;
		oss << "invalid cleanup_reel_key: " << s;
		*error = oss.str();
	}
	return false;
}

}  // namespace

bool loadAppConfig(const std::string& path, AppConfig* outConfig, std::string* error) {
	if (outConfig == nullptr) {
		if (error != nullptr) {
			*error = "config output pointer is null";
		}
		return false;
	}

	ZIni ini(path.c_str());
	AppConfig config{};

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
	config.cleanup_reel_key_name = ini.get("vrchat_fish", "reel_key", "T");
	if (!parseKeyVal(config.cleanup_reel_key_name, &config.cleanup_reel_key, error)) {
		return false;
	}
	const int legacyLootClickDelayMs = ini.getInt("vrchat_fish", "loot_click_delay_ms", 200);
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

	std::cout << "Loaded VRChat config: window_class=" << config.window_class
		<< ", title_contains=" << config.window_title_contains
		<< ", target=" << config.target_width << "*" << config.target_height << std::endl;
	if (config.ml_mode == 1) {
		std::cout << "[ML] Record mode enabled, csv: " << config.ml_record_csv << std::endl;
	} else if (config.ml_mode == 2) {
		std::cout << "[ML] Inference mode enabled, weights: " << config.ml_weights_file << std::endl;
	}
	*outConfig = config;
	return true;
}

AppConfig loadAppConfig(const std::string& path) {
	AppConfig config{};
	std::string error;
	if (!loadAppConfig(path, &config, &error)) {
		std::cout << "[config] load failed: " << error << std::endl;
	}
	return config;
}
