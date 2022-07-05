#pragma once
#define PI 3.14159265
#include <math.h>

class TrigonometryCalc
{
public:
	TrigonometryCalc()
	{
		for (int i = 0; i < 360; i++)
		{
			float radian = float(i * PI / 180);
			sinLook[i] = sin(radian);
			cosLook[i] = cos(radian);
		}
	}
	float getSin(int degree)
	{
		if (degree >= 360)
		{
			degree %= 360;
			return sinLook[degree];
		}
		else if (degree < 0)
		{
			return sinLook[360 + degree];
		}
		else
		{
			return sinLook[degree];
		}
	}
	float getCos(int degree)
	{
		if (degree >= 360)
		{
			degree %= 360;
			return cosLook[degree];
		}
		else if (degree < 0)
		{
			return cosLook[360 + degree];
		}
		else
		{
			return cosLook[degree];
		}
	}

private:
	float cosLook[360];
	float sinLook[360];
};