
***Ubuntu  cleanup***
#see file size 
`sudo du -h --max-depth 2 /home/username`

https://www.geeksforgeeks.org/du-command-linux-examples/

`python --version`

#set python version globally
`alias python=python3`

#check it
`which python`


#setup venv command
`python -m venv env`

***create project folder & start it***

mkdir tutorial
cd tutorial
`python -m venv env`

ls env/
`source env/bin/activate`


***debug code***
 `import pdb; pdb.set_trace()`
 #cheat sheet #
https://github.com/nblock/pdb-cheatsheet/releases
https://www.youtube.com/watch?v=BixeKmlKOJc


 # c to continue 
  # n to next step 
  # q to quit 

#install requiremtns
`pip install -r requirements.txt`

#list all packages
`pip list`

# freeze
`pip freeze`


#shutdown
`deactivate`








