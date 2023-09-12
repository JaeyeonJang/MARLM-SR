# MARLM-SR
This is code page for paper "Learning Multiple Coordinated Agents under Directed Acyclic Graph Constraints" submitted to ICLR 2024.
For confidentiality reasons, we offer only the following three artificial environments in our repository.

<div align="center">

<img src = "https://github.com/n2kdnk1123/MARLM-SR/assets/103877745/e79f4cf9-3523-4b59-804d-432bdb3a09fd" width="10%" height="10%">

**Factory production planning**

</div>

<div align="center">

![logistics](https://github.com/n2kdnk1123/MARLM-SR/assets/103877745/98ab541e-a22b-44d4-a2f6-c8ecf9a4591b)

**Logistics**

</div>

<div align="center">

![HPP](https://github.com/n2kdnk1123/MARLM-SR/assets/103877745/11f2210e-525e-4b96-9364-4d2bbf8ab537)

**Hierarchical predator-prey**

</div>



The code is based on the [ray/rllib framework](https://docs.ray.io/en/latest/rllib/index.html) with the support of the Python/TensorFlow framework.

## Environments
You can find the details of the three implemented environments in the Appendix of the submitted paper.

### Factory production planning
Set the model and environmental settings in **Factory_production_planning.py**, and then run this Python file.

### Logistics
Set the model and environmental settings in **Logistics.py**, and then run this Python file.

### Hierarchical predator-prey
Set the model and environmental settings in **Hierarchical_predator_prey.py**, and then run this Python file.

## Code structure
In **multiagent/scenarios**: the basic components of each environment are defined.

**multiagent/core_{environment name}.py**: defines agents and environment.

**multiagent/env_{environment name}.py**: provides environments for the main algorithm MARLM-SR based on ray/rllib framework.

**multiagent/scenario_{environment name}.py**: provides basic functions for environments.

In **rllib_mod**: trainer is defined based on proximal policy optimization algorithm.

In **utils**: logger is defined to record learning curve based on reward.


