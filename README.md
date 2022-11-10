Here is a list of commands you can run for the project to run. This was tested on windows best should work in other operating systems as well.

cd into the source folder of the project. Then,

'venv\Scripts\activate'
Create a python virtual environment

If you get an error saying you are not allowed to run scripts. Run the below command.
'Set-ExecutionPolicy Unrestricted -Force'

'python Run.py' 
Run the live camera happy detection

'python RunSingle.py' 
This command runs the detector on the file you select from the input_files folder