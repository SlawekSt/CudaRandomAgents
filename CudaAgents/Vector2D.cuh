#pragma once
#include <math.h>
#include "SFML/Graphics.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

class Vector2D
{
public:
	__host__ __device__ Vector2D();
	__host__ __device__ Vector2D(float x, float y);
	__host__ __device__ Vector2D(sf::Vector2f vec);

	template<typename T>
	__host__ __device__ void add(T vec);
	template<typename T>
	__host__ __device__ void sub(T vec);

	__host__ __device__ void mult(float n);
	__host__ __device__ void div(float n);
	__host__ __device__ float mag();
	__host__ __device__ float dot(Vector2D vec);
	__host__ __device__ void normalize();

	__host__ __device__ static Vector2D add(Vector2D vec1, Vector2D vec2);
	__host__ __device__ static Vector2D sub(Vector2D vec1, Vector2D vec2);
	__host__ __device__ static float dist(Vector2D vec1, Vector2D vec2);
	__host__ static Vector2D limit(Vector2D vec1, float maxValue);
	__device__ static Vector2D Climit(Vector2D vec1, float maxValue)
	{
		float n = vec1.mag();
		if (n == 0)
		{
			return Vector2D(0.0f, 0.0f);
		}
		float f = fminf(n, maxValue) / n;
		return Vector2D(f * vec1.x, f * vec1.y);
	}

	__host__ __device__ operator sf::Vector2f() const;

public:
	float x;
	float y;
};

inline Vector2D::Vector2D()
{
	x = 0.f;
	y = 0.f;
}

inline Vector2D::Vector2D(float x, float y)
{
	this->x = x;
	this->y = y;
}

inline Vector2D::Vector2D(sf::Vector2f vec)
{
	x = vec.x;
	y = vec.y;
}

inline void Vector2D::mult(float n)
{
	x *= n;
	y *= n;
}

inline void Vector2D::div(float n)
{
	x /= n;
	y /= n;
}

inline float Vector2D::mag()
{
	return sqrt(x * x + y * y);
}

inline float Vector2D::dot(Vector2D vec)
{
	return x * vec.x + y * vec.y;
}

inline void Vector2D::normalize()
{
	float m = mag();
	if (m != 0)
	{
		div(m);
	}
}

inline Vector2D Vector2D::add(Vector2D vec1, Vector2D vec2)
{
	Vector2D newVec(vec1.x + vec2.x, vec1.y + vec2.y);
	return newVec;
}

inline Vector2D Vector2D::sub(Vector2D vec1, Vector2D vec2)
{
	Vector2D newVec(vec1.x - vec2.x, vec1.y - vec2.y);
	return newVec;
}

inline float Vector2D::dist(Vector2D vec1, Vector2D vec2)
{
	return static_cast<float>(sqrt(pow(vec1.x - vec2.x, 2) + pow(vec1.y - vec2.y, 2)));
}

inline Vector2D Vector2D::limit(Vector2D vec1, float maxValue)
{
	float n = vec1.mag();
	if (n == 0)
	{
		return Vector2D(0.0f, 0.0f);
	}
	float f = std::min(n, maxValue) / n;
	return Vector2D(f * vec1.x, f * vec1.y);
}

inline Vector2D::operator sf::Vector2f() const
{
	return sf::Vector2f(x, y);
}

template<typename T>
inline void Vector2D::add(T vec)
{
	x += vec.x;
	y += vec.y;
}

template<typename T>
inline void Vector2D::sub(T vec)
{
	x -= vec.x;
	y -= vec.y;
}