#include <iostream>
#include <thread>

#include <windows.h>

#include "engine/fish_engine.h"
#include "runtime/runtime_context.h"

#define KEY_PRESSED(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x0001) ? 1 : 0)

int main() {
	runtime::RuntimeContext runtime;
	if (!runtime.initialize("config.ini")) {
		return 1;
	}

	std::cout << std::endl;
	engine::FishEngine fishEngine(runtime);

	if (runtime.config().is_pause) {
		std::cout << "Press Tab to pause/resume" << std::endl;
		std::thread worker([&fishEngine]() {
			fishEngine.runLoop();
		});
		worker.detach();

		while (true) {
			if (KEY_PRESSED(VK_TAB)) {
				fishEngine.togglePause();
				if (fishEngine.isPaused()) {
					std::cout << "Paused" << std::endl;
				} else {
					std::cout << "Resumed" << std::endl;
				}
			}
			Sleep(500);
		}
	} else {
		fishEngine.runLoop();
	}

	return 0;
}
