#pragma once
#include "EntityManager.h"
#include "Wanderer.h"
#include "TrignometryCalc.h"


class AgentManager : public EntityManager
{
public:
	AgentManager();
	void draw(sf::RenderTarget& target) override;
	void update() override;
	void updateAgent() override;
private:
	std::vector<Wanderer> agents;
	const int agentsNumber = 1'000'000;
	sf::VertexArray agentsVertex;
	static TrigonometryCalc triCalc;
};
