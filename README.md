This TensorFlow version of DDPG has been written during the master's thesis of Arnaud de Froissard de Broissia, under the supervision of ![Olivier Sigaud](http://www.isir.upmc.fr/index.php?op=view_profil&lang=en&id=28) (Olivier.Sigaud@upmc.fr)

It was used to obtain the results described in the following paper:
http://arxiv.org/abs/1606.09152

It has been coded under strong time constraints, thus the code is "quick and dirty". Maybe you should consider using more advanced versions of DDPG available on gitHub before using this one.

To install this version of DDPG (two methods):

First method: 

	1)Clone repository somewhere.

	2)add to your .bashrc file : export PYTHONPATH=$PYTHONPATH:(path of the DDPG directory's parent)

Second method:

	1)In a terminal type "echo $PYTHONPATH"

	2)Clone the repository to the directory indicated by PYTHONPATH

Test if it worked:

	1)open a python terminal

	2)type :"import DDPG.test.test_mc as mc"

	3)[optional] type :"mc.doEp(100)"
