Project is still under construction, but should work in windows

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

Below link provides details of this projects working. It describes the same algorithm detecting all emotions, but, this one focuses on just happy. ENMimages folder of the above project uses the cropped images of eyes, nose, mouth, and their landmarks together to make predictions.

https://medium.com/@rishiswethan.c.r/emotion-detection-using-facial-landmarks-and-deep-learning-b7f54fe551bf

Here is a video of the same algotihm using landmarks and images to make predictions

https://www.youtube.com/watch?v=H5aaYGRGxDo
