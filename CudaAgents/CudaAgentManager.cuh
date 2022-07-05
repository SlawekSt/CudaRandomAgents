#pragma once
#include "EntityManager.h"
#include "CudaWanderer.cuh"
#include "TrignometryCalc.h"

class CudaAgentManager : public EntityManager
{
public:
	CudaAgentManager();
	~CudaAgentManager();
	void draw(sf::RenderTarget& target) override;
	void update() override;
	void updateAgent() override;
private:
	const int agentsNumber = 1'000'000;
	sf::VertexArray agentsVertex;
	CudaWanderer* agents;
	// Data to draw
	std::vector<Vector2D> positions;
	std::vector<float> angles;
	std::vector<float> randomNumbers;
	// Cuda counterpart
	Vector2D* resPos;
	float* resAngle;
	float* resRandom;
	static TrigonometryCalc triCalc;
};