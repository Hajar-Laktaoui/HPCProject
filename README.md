Ce projet propose une simulation 1D de rupture de barrage à l’aide du schéma de Rusanov, avec deux versions parallèles :

OpenMP (mémoire partagée) : dam_break_rusanov_openmp.c

MPI (mémoire distribuée) : dam_break_rusanov_mpi.c

**Compilation : **
Pour OpenMP: gcc -fopenmp dam_break_rusanov_openmp.c -o dam_openmp
Pour MPI : mpicc dam_break_rusanov_mpi.c -o dam_mpi

**Exécution : **
Pour OpenMP : OMP_NUM_THREADS=4 ./dam_openmp
Pour MPI : mpirun -np 4 ./dam_mpi

==> Les résultats présentés dans le rapport ont été obtenus en utilisant le framework HPC Toubkal.

