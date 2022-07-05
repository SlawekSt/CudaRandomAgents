#include "CudaWanderer.cuh"
#define PI 3.14159265
#include <iostream>

constexpr int threadNumber = 32;

CudaWanderer::CudaWanderer(Vector2D startingPos, float angle) : position(startingPos), angle(angle)
{

}

__device__ void CudaWanderer::update(float maxSpeed) noexcept
{
	//printf("Acceleration %f - %f\n", acceleration.x, acceleration.y);
	//printf("Velocity %f - %f\n", velocity.x, velocity.y);
	velocity.add(acceleration);
	// Velocity cant be higher than MaxSpeed
	velocity = Vector2D::Climit(velocity, maxSpeed);
	// Reset acceleration
	acceleration.mult(0);
}

__device__ void CudaWanderer::updatePosition() noexcept
{
	position.add(velocity);
	angle = static_cast<float>((atan2(velocity.y, velocity.x) * 180 / PI)) + 360;
	angle = static_cast<float>(int(angle) % 360);
}

__device__ void CudaWanderer::applayForce(Vector2D force) noexcept
{
	acceleration.add(force);
}

__device__ void CudaWanderer::wander(float maxSpeed, float maxForce, float wanderDistance, float wanderRadius, float randomAngle, Vector2D simulationBound)
{
	// Locate middle point of wandering circle
	Vector2D circlePos = velocity;
	circlePos.normalize();
	// How far from agent is wandering circle
	circlePos.mult(wanderDistance);
	// Check if circle collide with simulation bounds
	Vector2D wallBound = circlePos;
	wallBound.add(position);
	// If collide change direction
	if (wallBound.x > simulationBound.x || wallBound.x < 0.0f)
		circlePos.x *= -1;
	if (wallBound.y > simulationBound.y || wallBound.y < 0.0f)
		circlePos.y *= -1;
	circlePos.add(position);
	float h = atan2(velocity.y, velocity.x);
	// Pick random point on wandering circle and seek toward it
	Vector2D circleOffSet = Vector2D(wanderDistance * cos(randomAngle + h), wanderRadius * sin(randomAngle + h));
	circlePos.add(circleOffSet);
	seek(circlePos,maxSpeed,maxForce);
}

__device__ void CudaWanderer::seek(Vector2D target,float maxSpeed,float maxForce)
{
	target.sub(position);
	target.normalize();
	target.mult(maxSpeed);
	target.sub(velocity);
	target = Vector2D::Climit(target, maxForce);
	applayForce(target);
}



__global__ void updateAgentPos(CudaWanderer* agents,Vector2D* positions,float* angles, unsigned size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size)
	{
		agents[tid].updatePosition();
		positions[tid] = agents[tid].getPosition();
		angles[tid] = agents[tid].getAngle();
	}
}

__global__ void updateAgent(CudaWanderer* agents, unsigned size, float maxSpeed, float maxForce, float wanderDistance, float wanderRadius, float* randomAngle, Vector2D simulationBound)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < size)
	{
		agents[tid].wander(maxSpeed, maxForce, wanderDistance, wanderRadius, randomAngle[tid], simulationBound);
		agents[tid].update(maxSpeed);
	}
}


void updateCudaAgentsPos(CudaWanderer* agents, Vector2D* positions, float* angles, unsigned size)
{
	updateAgentPos << <(size + threadNumber - 1)/threadNumber, threadNumber >> > (agents, positions, angles, size);
}

void updateCudaAgent(CudaWanderer* agents, unsigned size, float maxSpeed, float maxForce, float wanderDistance, float wanderRadius, float* randomAngle, Vector2D simulationBound)
{
	updateAgent << <(size + threadNumber - 1) / threadNumber, threadNumber >> > (agents, size, maxSpeed,maxForce,wanderDistance,wanderRadius,randomAngle,simulationBound);
}
