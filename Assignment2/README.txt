To run the code, you must have ABAGAIL and JAVA, and ANT installed. A copy of the edited ABAGAIL folder is included with this project file. Ant can be installed with homebrew on MAC or any equivalent in PC. 

To run the code for each algorithm, first navigate to the folder ABAGAIL-master in your terminal: 
		1) Compile the code with 'ant'

		For part 1 run the following command:
		1) java -cp ABAGAIL.jar opt.test.AbaloneTest

Certain regions of the code may be commented out. Please refer to the comments to get the results found in this paper.

		For part 2, 
		2) java -cp ABAGAIL.jar opt.test.TravelingSalesmanTest
		3) java -cp ABAGAIL.jar opt.test.KnapsackTest
		4) java -cp ABAGAIL.jar opt.test.FlipFlopTest


The GoT dataset is found in the test file found with the path: src/opt/test. 
You may uncomment by removing the '//' before each line and rerun the code to see the data points. 
The raw data for each diagram is also available to see in the xlsx file named 'rawdata'.