#include "AgentManager.h"
#include "RandomNumberGenerator.h"
#include "TrignometryCalc.h"
#include <omp.h>
#include <chrono>

AgentManager::AgentManager() : agentsVertex(sf::Triangles, agentsNumber * static_cast<size_t>(3))
{
	Vector2D startingPos(1000.0f, 1000.0f);
	for (int i = 0; i < agentsNumber; ++i)
	{
		agents.emplace_back(startingPos, RandomNumberGenerator::randFloat(0.0f, 360.0f));
	}

	for (int i = 0; i < agentsNumber * 3; ++i)
	{
		agentsVertex[i].color = sf::Color::Green;
	}
}

void AgentManager::draw(sf::RenderTarget& target)
{
	auto t_start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < agentsNumber; ++i)
	{
		float angle = agents[i].getAngle();
		Vector2D position = agents[i].getPosition();
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

void AgentManager::update()
{
	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < agentsNumber; ++i)
	{
		agents[i].wander();
		agents[i].update();
	}
}

void AgentManager::updateAgent()
{
	#pragma omp parallel for num_threads(6)
	for (int i = 0; i < agentsNumber; ++i)
	{
		agents[i].updatePosition();
	}
}

TrigonometryCalc AgentManager::triCalc;
