#pragma once
#include "Vector2D.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

class CudaWanderer
{
public:
	__host__ CudaWanderer(Vector2D startingPos, float angle);
	__device__ void update(float maxSpeed) noexcept;
	__device__ void updatePosition() noexcept;
	__device__ void applayForce(Vector2D force) noexcept;
	// Getters
	__device__ Vector2D getPosition() { return position; }
	__device__ float getAngle() { return angle; }
	// Movement
	__device__ void wander(float maxSpeed, float maxForce,float wanderDistance,float wanderRadius, float randomAngle, Vector2D simulationBound);
	__device__ void seek(Vector2D target, float maxSpeed, float maxForce);
private:
	Vector2D position;
	Vector2D velocity;
	Vector2D acceleration;
	float angle;
};


void updateCudaAgentsPos(CudaWanderer* agents, Vector2D* positions, float* angles, unsigned size);
void updateCudaAgent(CudaWanderer* agents, unsigned size, float maxSpeed, float maxForce, float wanderDistance, float wanderRadius, float* randomAngle, Vector2D simulationBound);