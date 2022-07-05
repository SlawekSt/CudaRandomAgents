#include "AgentSimulation.h"
#include <chrono>
#include <vector>
#include <omp.h>
int main()
{
	AgentSimulation sim;
	sim.run();

	return 0;
}