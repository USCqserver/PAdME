# PAdME: Time-Partitioned Adiabatic Master Equation Solver

Package for integrating the Markovian Adiabatic Master Equation 
by constructing time interval partitions and integrating over 
a fixed truncated energy basis on each partition. This drastically
reduces the numerical effort required to diagonalize on each integration step.
This method compresses and throws away high energy levels, 
but trace loss can be mitigated with finer partitions as long as
the evolution is adiabatic.

Requires Qutip, numpy and scipy. Currently under early development. 
Comprehensive testing and documentation to come. 
No warranties whatsover. 