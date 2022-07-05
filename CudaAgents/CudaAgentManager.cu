#include "CudaAgentManager.cuh"
#include "RandomNumberGenerator.h"
#include "AntSettings.h"
#include <chrono>
#include "Drawer.h"
#include <omp.h>

CudaAgentManager::CudaAgentManager() : agentsVertex(sf::Triangles, agentsNumber* static_cast<size_t>(3))
{
	std::vector<CudaWanderer> agentsVec;
	agentsVec.reserve(agentsNumber);
	Vector2D startingPos(1000.0f, 1000.0f);
	float angle{};
	positions.reserve(agentsNumber);
	angles.reserve(agentsNumber);
	randomNumbers.reserve(agentsNumber);
	for (int i = 0; i < agentsNumber; ++i)
	{
		angle = RandomNumberGenerator::randFloat(0.0f, 360.0f);
		agentsVec.emplace_back(startingPos, angle);
		angles.push_back(angle);
		positions.push_back(startingPos);
		randomNumbers.push_back(RandomNumberGenerator::randFloat(0, 6.2831f));
	}

	for (int i = 0; i < agentsNumber * 3; ++i)
	{
		agentsVertex[i].color = sf::Color::Green;
	}

	//Reserve memory on device
	cudaMalloc(&agents, sizeof(CudaWanderer) * agentsNumber);
	cudaMalloc(&resPos, sizeof(Vector2D) * agentsNumber);
	cudaMalloc(&resAngle, sizeof(float) * agentsNumber);
	cudaMalloc(&resRandom, sizeof(float) * agentsNumber);

	cudaMemcpy(agents, agentsVec.data(), sizeof(CudaWanderer) * agentsNumber, cudaMemcpyHostToDevice);
	cudaMemcpy(resRandom, randomNumbers.data(), sizeof(float) * agentsNumber, cudaMemcpyHostToDevice);
}

CudaAgentManager::~CudaAgentManager()
{
	cudaFree(agents);
	cudaFree(resPos);
	cudaFree(resAngle);
	cudaFree(resRandom);
}

void CudaAgentManager::draw(sf::RenderTarget& target)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < agentsNumber; ++i)
	{
		float angle = angles[i];
		Vector2D position = positions[i];
		int currentAngle = static_cast<int>(angle);
		agentsVertex[i * 3].position = sf::Vector2f(position.x + 10 * triCalc.getCos(currentAngle), position.y + 10 * triCalc.getSin(currentAngle));
		currentAngle = static_cast<int>(angle) - 90;
		agentsVertex[i * 3 + 1].position = sf::Vector2f(position.x + 6 * triCalc.getCos(currentAngle), position.y + 6 * triCalc.getSin(currentAngle));
		currentAngle = static_cast<int>(angle) + 90;
		agentsVertex[i * 3 + 2].position = sf::Vector2f(position.x + 6 * triCalc.getCos(currentAngle), position.y + 6 * triCalc.getSin(currentAngle));
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	std::cout << "Draw UP: " << elapsed_time_ms << std::endl;
	target.draw(agentsVertex);
}

void CudaAgentManager::update()
{
	updateCudaAgent(agents, agentsNumber, AntSettings::getInstance().MaxSpeed(), AntSettings::getInstance().MaxForce(), AntSettings::getInstance().WanderDistance(), AntSettings::getInstance().WanderRadius(), resRandom, AntSettings::getInstance().SimulationBound());
	// Update device agents and get results
	updateCudaAgentsPos(agents, resPos, resAngle, agentsNumber);
}

void CudaAgentManager::updateAgent()
{
	// Copy result from device to host
	cudaMemcpy(positions.data(), resPos, sizeof(Vector2D) * agentsNumber, cudaMemcpyDeviceToHost);
	cudaMemcpy(angles.data(), resAngle, sizeof(float) * agentsNumber, cudaMemcpyDeviceToHost);
	
	// Reset randomNumbers for next frame
	getRandomNumbers(randomNumbers, agentsNumber, 0.0f, 6.2831f);

	cudaMemcpy(resRandom, randomNumbers.data(), sizeof(float) * agentsNumber, cudaMemcpyHostToDevice);
}

TrigonometryCalc CudaAgentManager::triCalc;
