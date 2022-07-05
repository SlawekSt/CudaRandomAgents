#pragma once
#include "AntSettings.h"

class Wanderer
{
public:
	Wanderer(Vector2D startingPos, float angle);
	void update() noexcept;
	void updatePosition() noexcept;
	void applayForce(Vector2D force) noexcept;
	// Getters
	Vector2D getPosition() { return position; }
	float getAngle() { return angle; }
	// Movement
	void wander();
	void seek(Vector2D target);

private:
	Vector2D position;
	Vector2D velocity;
	Vector2D acceleration;
	float angle;
};

