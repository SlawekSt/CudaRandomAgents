#pragma once
#include "ResourcesManager.h"
#include "Vector2D.cuh"

class AntSettings
{
public:
	AntSettings()
	{
		ResourcesManager manager("AntConfig.lua");

		maxSpeed = manager.getFloat("MaxSpeed");
		maxForce = manager.getFloat("MaxForce");
		wanderDistance = manager.getFloat("WanderDistance");
		wanderRadius = manager.getFloat("WanderRadius");
		simulationBound = Vector2D(manager.getFloat("BoundX"), manager.getFloat("BoundY"));
	}

	static AntSettings& getInstance()
	{
		static AntSettings instance;
		return instance;
	}

	AntSettings(AntSettings const&) = delete;
	void operator=(AntSettings const&) = delete;

	float MaxSpeed() const { return maxSpeed; }
	float MaxForce() const { return maxForce; }
	float WanderDistance() const { return wanderDistance; }
	float WanderRadius() const { return wanderRadius; }
	Vector2D SimulationBound() const { return simulationBound; }

private:
	float maxSpeed;
	float maxForce;
	float wanderDistance;
	float wanderRadius;
	Vector2D simulationBound;
};