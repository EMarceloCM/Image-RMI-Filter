# how to run it:
<br> sudo apt update <br>
sudo apt install -y libopenmpi-dev openmpi-bin python3-mpi4py <br>
mpirun -np 5 --hostfile t.txt python3 ex01.py <br>
mpirun -np 5 --hostfile t.txt python3 ex03.py
