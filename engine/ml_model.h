#pragma once

#include <string>
#include <vector>

namespace engine {

struct MlpLayer {
	int in_dim{};
	int out_dim{};
	std::vector<double> weights;
	std::vector<double> bias;
};

struct MlpModel {
	std::vector<MlpLayer> layers;
	std::vector<double> norm_mean;
	std::vector<double> norm_std;
	bool loaded = false;
};

bool loadMlpWeights(const std::string& path, MlpModel& model);

}  // namespace engine
