#!/bin/bash
###PBS -N HW3_Math529
###PBS -W group_list=mmo
###PBS -q standard 
###PBS -m bea
###PBS -M jesseadams@email.arizona.edu
###PBS -l select=1:ncpus=1:mem=6gb:pcmem=6gb 
###PBS -l walltime=00:05:00 
###PBS -l cput=00:05:00 

module load python

cd ~jesseadams/DandI/Homework3
date

### rga=$(seq 1 100)
### rgr=$(seq 0 30)
rga=$(seq 10 11)
rgr=$(seq 10 11)

for a in ${rga}
do
	for r in ${rgr}
	do
		echo "Submitting job Homework3.main( ${a}/1e2, ${r}/1e1 )"
		python3 -c "import Homework3; Homework3.main( ${a}/1e2, ${r}/1e1 )" &
	done
done

date
