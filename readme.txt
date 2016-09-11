To install DDPG (two ways):

First method: 
	1) Clone the repository somewhere.
	2) add to your .bashrc file : export PYTHONPATH=$PYTHONPATH:(path of the DDPG directory's parent)

Second method:
	1) In a terminal type "echo $PYTHONPATH"
	2) Clone the repository to the directory indicated by $PYTHONPATH

Test if it worked :
	1) open a python terminal
	2) type :"import DDPG.test.test_mc as mc"
	3) [optional] type :"mc.doEp(100)"
