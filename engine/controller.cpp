#include "controller.h"

#include <algorithm>
#include <cmath>

namespace engine {

ControlDecision computeControlDecision(const ControlInput& input, const AppConfig& config) {
	ControlDecision decision{};
	decision.wantHold = input.holding;
	decision.reactiveTriggered = false;
	decision.deviationForNext = input.prevDeviation;
	decision.hasDeviationForNext = input.hasPrevDeviation;

	double gravity = config.bb_gravity;
	double thrust = config.bb_thrust;
	double drag = config.bb_drag;
	if (drag < 0.5) drag = 0.5;
	if (drag > 1.0) drag = 1.0;
	int horizon = config.bb_sim_horizon;
	if (horizon < 1) horizon = 1;
	if (horizon > 30) horizon = 30;

	double marginRatio = config.bb_margin_ratio;
	if (marginRatio < 0.0) marginRatio = 0.0;
	if (marginRatio > 0.45) marginRatio = 0.45;
	double margin = static_cast<double>(input.sliderHeight) * marginRatio;

	double fishVelDecay = config.fish_vel_decay;
	if (fishVelDecay < 0.5) fishVelDecay = 0.5;
	if (fishVelDecay > 1.0) fishVelDecay = 1.0;
	const bool fishBounce = (config.fish_bounce_predict != 0);

	auto simCost = [&](bool press) -> double {
		double sVel = input.smoothSliderVelocity;
		double sCY = static_cast<double>(input.sliderCenterY);
		double fY = static_cast<double>(input.fishY);
		double fVel = input.smoothFishVelocity;
		double fAcc = input.smoothFishAccel;
		double cost = 0.0;

		double trackPadY = static_cast<double>(std::max(0, config.track_pad_y));
		double trackCYMin = static_cast<double>(input.fixedTrackRoi.y) + trackPadY;
		double trackCYMax = static_cast<double>(input.fixedTrackRoi.y + input.fixedTrackRoi.height) - trackPadY;
		if (trackCYMin > trackCYMax) {
			double mid = (trackCYMin + trackCYMax) / 2.0;
			trackCYMin = mid;
			trackCYMax = mid;
		}

		for (int step = 0; step < horizon; step++) {
			double accel = press ? thrust : gravity;
			sVel = sVel * drag + accel;
			sCY += sVel;

			if (sCY < trackCYMin) {
				sCY = trackCYMin;
				sVel = -sVel * 0.8;
			} else if (sCY > trackCYMax) {
				sCY = trackCYMax;
				sVel = -sVel * 0.8;
			}

			fVel += fAcc;
			fVel *= fishVelDecay;
			fY += fVel;

			if (fishBounce) {
				if (fY < trackCYMin) {
					fY = trackCYMin;
					fVel = -fVel * 0.5;
					fAcc = 0.0;
				} else if (fY > trackCYMax) {
					fY = trackCYMax;
					fVel = -fVel * 0.5;
					fAcc = 0.0;
				}
			}

			double halfH = static_cast<double>(input.sliderHeight) / 2.0;
			double sTop = sCY - halfH + margin;
			double sBot = sCY + halfH - margin;

			if (sTop >= sBot) {
				cost += std::abs(fY - sCY);
			} else if (fY < sTop) {
				cost += (sTop - fY);
			} else if (fY > sBot) {
				cost += (fY - sBot);
			}
		}

		double bZone = config.bb_boundary_zone;
		double bWeight = config.bb_boundary_weight;
		if (bZone > 0.0 && bWeight > 0.0) {
			double trackPadYLocal = static_cast<double>(std::max(0, config.track_pad_y));
			double trackCYMinLocal = static_cast<double>(input.fixedTrackRoi.y) + trackPadYLocal;
			double trackCYMaxLocal = static_cast<double>(input.fixedTrackRoi.y + input.fixedTrackRoi.height) - trackPadYLocal;
			double distTop = sCY - trackCYMinLocal;
			double distBot = trackCYMaxLocal - sCY;
			double distMin = (distTop < distBot) ? distTop : distBot;
			if (distMin < bZone) {
				double penetration = (bZone - distMin) / bZone;
				cost += std::abs(sVel) * penetration * bWeight;
			}
		}

		return cost;
	};

	decision.costPress = simCost(true);
	decision.costRelease = simCost(false);

	if (std::abs(decision.costPress - decision.costRelease) < 0.5) {
		decision.wantHold = input.holding;
	} else {
		decision.wantHold = (decision.costPress < decision.costRelease);
	}

	if (config.reactive_override) {
		double deviation = static_cast<double>(input.fishY - input.sliderCenterY);
		double absDev = std::abs(deviation);
		double devThreshold = static_cast<double>(input.sliderHeight) * config.reactive_dev_ratio;
		if (devThreshold < 10.0) {
			devThreshold = 10.0;
		}

		if (absDev > devThreshold && input.hasPrevDeviation) {
			double devGrowth = (absDev - std::abs(input.prevDeviation)) / input.lastDtRatio;
			if (devGrowth > config.reactive_grow_threshold) {
				bool reactiveHold = (deviation < 0.0);
				if (reactiveHold != decision.wantHold) {
					decision.wantHold = reactiveHold;
					decision.reactiveTriggered = true;
				}
			}
		}

		decision.deviationForNext = deviation;
		decision.hasDeviationForNext = true;
	}

	return decision;
}

}  // namespace engine
