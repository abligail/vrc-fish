#pragma once

#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "../core/types.h"
#include "ml_model.h"

namespace runtime {
class RuntimeContext;
}

namespace engine {

class FishEngine {
public:
	explicit FishEngine(runtime::RuntimeContext& runtimeContext);

	void runLoop();
	void togglePause();
	bool isPaused() const;

private:
	struct LoopState;

	unsigned long long nowMs() const;
	void writeVrLogLine(LoopState& loop, const std::string& line, bool alsoStdout = true) const;
	void switchState(LoopState& loop, VrFishState next) const;
	void sleepWithPause(LoopState& loop, int totalMs) const;
	void cleanupToNextRound(LoopState& loop, const std::string& tag) const;

	void runCastStep(LoopState& loop) const;
	void runEnterMinigameStep(LoopState& loop) const;
	void runWaitBiteStep(LoopState& loop, const cv::Mat& frame, const cv::Mat& gray) const;
	void runControlMinigameStep(LoopState& loop, const cv::Mat& frame, const cv::Mat& gray) const;
	void runPostMinigameStep(LoopState& loop, const cv::Mat& frame) const;

	std::string makeDebugPath(const std::string& tag) const;
	void saveDebugFrame(const cv::Mat& bgr, const std::string& tag) const;
	void saveDebugFrame(const cv::Mat& bgr, const std::string& tag, const cv::Rect& r1,
		const cv::Scalar& c1 = cv::Scalar(0, 0, 255)) const;
	void saveDebugFrame(const cv::Mat& bgr, const std::string& tag, const cv::Rect& r1, const cv::Rect& r2) const;
	void saveDebugFrame(const cv::Mat& bgr, const std::string& tag, const cv::Rect& r1,
		const cv::Rect& r2, const cv::Rect& r3) const;

	runtime::RuntimeContext& runtime_;
	MlpModel mlpModel_{};
};

}  // namespace engine
