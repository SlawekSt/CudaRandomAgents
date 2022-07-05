#include "Wanderer.h"
#include "AntSettings.h"
#include "RandomNumberGenerator.h"
constexpr float PI = 3.14159265f;
const int threadNumber = 128;


Wanderer::Wanderer(Vector2D startingPos, float angle) : position(startingPos),angle(angle)
{
}

void Wanderer::update() noexcept
{
	velocity.add(acceleration);
	// Velocity cant be higher than MaxSpeed
	velocity = Vector2D::limit(velocity, AntSettings::getInstance().MaxSpeed());
	// Reset acceleration
	acceleration.mult(0);
}

void Wanderer::updatePosition() noexcept
{
	position.add(velocity);
	angle = static_cast<float>((atan2(velocity.y, velocity.x) * 180 / PI)) + 360;
	angle = static_cast<float>(int(angle) % 360);
}

void Wanderer::applayForce(Vector2D force) noexcept
{
	acceleration.add(force);
}

void Wanderer::wander()
{
	float theta = RandomNumberGenerator::randFloat(0, 6.2831f);
	// Locate middle point of wandering circle
	Vector2D circlePos = velocity;
	circlePos.normalize();
	// How far from agent is wandering circle
	circlePos.mult(AntSettings::getInstance().WanderDistance());
	// Check if circle collide with simulation bounds
	Vector2D wallBound = circlePos;
	wallBound.add(position);
	// If collide change direction
	if (wallBound.x > AntSettings::getInstance().SimulationBound().x || wallBound.x < 0.0f)
		circlePos.x *= -1;
	if (wallBound.y > AntSettings::getInstance().SimulationBound().y || wallBound.y < 0.0f)
		circlePos.y *= -1;
	circlePos.add(position);

	float h = atan2(velocity.y, velocity.x);
	// Pick random point on wandering circle and seek toward it
	Vector2D circleOffSet = Vector2D(AntSettings::getInstance().WanderDistance() * cos(theta + h), AntSettings::getInstance().WanderRadius() * sin(theta + h));
	circlePos.add(circleOffSet);
	seek(circlePos);
}

void Wanderer::seek(Vector2D target)
{
	target.sub(position);
	target.normalize();
	target.mult(AntSettings::getInstance().MaxSpeed());
	target.sub(velocity);
	target = Vector2D::limit(target, AntSettings::getInstance().MaxForce());
	applayForce(target);
}