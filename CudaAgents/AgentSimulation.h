#pragma once
#include "Simulation.h"
#include "CameraController.h"
#include "Wanderer.h"
#include "AgentManager.h"
#include "CudaAgentManager.cuh"

class AgentSimulation : public Simulation
{
public:
	AgentSimulation();
	void run() override;
private:
	void pollEvent() override;
	void update() override;
	void draw() override;
private:
	bool pause{ true };
	CameraController camera;
	CudaAgentManager manager;
};