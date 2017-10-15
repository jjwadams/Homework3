#!/bin/bash
### script to run a serial job using one core on htc using queue windfall or standard
### beginning of line, three pound/cross-hatch characters indicate comment
### beginning of line #PBS indicates an active PBS command/directive
### Set the job name
#PBS -N Geomag
### Specify the PI group for this job
### List of PI groups available to each user can be found with "va" command
#PBS -W group_list=mmo
### Set the queue for this job as windfall or standard (adjust ### and #)
#PBS -q standard
###PBS -q windfall
### Request email when job begins and ends - commented out in this case
#PBS -m bea
### Specify email address to use for notification - commented out in this case
# PBS -M jesseadams@email.arizona.edu



### Set the number of nodes, cores and memory that will be used for this job
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=6gb

### Important!!! Include this line for your 1 core job.
### Without it, the entire node, containing 12 cores, will be allocated
###PBS -l place=pack:free

### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=24:00:00
### Specify total cpu time required for this job, hhh:mm:ss
### total cputime = walltime * ncpus
#PBS -l cput=672:00:00
#
### Use "module avail" command to list all available modules
module load matlab
### set directory for job execution, ~netid = home directory path,
cd ~jesseadams/Geomagnetics
### run your executable program with begin and end date and time output
date


### a = 100
for a in {100..105..5}
do
	matlab -nodisplay -nosplash -nodesktop -r "a=$a/1e2; spanSm=1; testSmoothing" &
	matlab -nodisplay -nosplash -nodesktop -r "a=$a/1e2; spanSm=25; testSmoothing" &
	matlab -nodisplay -nosplash -nodesktop -r "a=$a/1e2; spanSm=50; testSmoothing" &
	matlab -nodisplay -nosplash -nodesktop -r "a=$a/1e2; spanSm=100; testSmoothing" &
done

date
