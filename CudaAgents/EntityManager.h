#pragma once
#include "SFML/Graphics.hpp"
class EntityManager
{
public:
	virtual void draw(sf::RenderTarget& target) = 0;
	virtual void update() = 0;
	// Second update function for agent position to allow parallel drawing and updating
	virtual void updateAgent() = 0;
};