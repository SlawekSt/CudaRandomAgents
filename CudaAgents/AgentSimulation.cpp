#include "AgentSimulation.h"
#include "ResourcesManager.h"
#include "RandomNumberGenerator.h"
#include <chrono>


AgentSimulation::AgentSimulation()
{
	ResourcesManager manager("WindowConfig.lua");

	window.create(sf::VideoMode(manager.getInt("WindowWidth"), manager.getInt("WindowHeight")), manager.getString("WindowTitle"));
	window.setFramerateLimit(manager.getInt("Framerate"));
}

void AgentSimulation::run()
{
	while (window.isOpen())
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		if (!pause)
		{
			update();
		}
		window.clear(sf::Color::White);
		draw();
		window.display();
		if (!pause)
		{
			manager.updateAgent();
		}
		pollEvent();
		auto t_end = std::chrono::high_resolution_clock::now();
		double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
		std::cout << "Frame: " << elapsed_time_ms << std::endl;
	}
}

void AgentSimulation::pollEvent()
{
	sf::Event e;
	while (window.pollEvent(e))
	{
		if (camera.handleWindowEvent(window, e))
		{
			continue;
		}
		if (e.type == sf::Event::Closed)
		{
			window.close();
			break;
		}
		if (e.type == sf::Event::KeyPressed)
		{
			if (e.key.code == sf::Keyboard::P)
			{
				pause = !pause;
			}
		}
	}
}

void AgentSimulation::update()
{
	manager.update();
}

void AgentSimulation::draw()
{
	auto t_start = std::chrono::high_resolution_clock::now();
	manager.draw(window);
	auto t_end = std::chrono::high_resolution_clock::now();
	double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	std::cout << "Draw: " << elapsed_time_ms << std::endl;
}
