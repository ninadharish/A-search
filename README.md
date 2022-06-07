![alt text](/output/plan.gif)

# Path Planning with the A* Algorithm

## Description

The [A* Algotihm](https://en.wikipedia.org/wiki/A*_search_algorithm) is used widely to traverse a graph between nodes and finding the optimal path. It is also extensively used for Robot Path Planning. For a given Obstacle Space, this program applies the A* Algorithm for Path Planning in order to find the optimal path for the robot to travel from a given initial potiion to a given goal position taking into consideration all the obstacles, the clearance space around them as well as the radius of the robot.

## Approach

* Pseudocode:

![alt text](/output/flo.png)

Source: Swift, Nicholas. “Easy A* (star) Pathfinding.” Medium, 27 February 2017, [link](https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2).

## Output

* The 2D Obstacle Space where the Obstacled are shown in red and a clearance space around the obstacles is shown in yellow.

![alt text](/output/obspc.png)

* Searching the path.

![alt text](/output/plan.gif)

* The Video Output can be found [here](https://drive.google.com/file/d/1amDYIq2HJnxtHP28z9022j47vOCS8FZB/view?usp=sharing).


## Getting Started

### Dependencies

<p align="left"> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>&ensp; </a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"> <img src="https://www.codebykelvin.com/learning/python/data-science/numpy-series/cover-numpy.png" alt="numpy" width="40" height="40"/>&ensp; </a>
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> <img src="https://avatars.githubusercontent.com/u/5009934?v=4&s=400" alt="opencv" width="40" height="40"/>&ensp; </a>

* [Python 3](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [OpenCV](https://opencv.org/)


### Executing program

* Clone the repository into any folder of your choice.
```
git clone https://github.com/ninadharish/Path-Planning-A-star-Algorithm.git
```

* Open the repository and navigate to the `src` folder.
```
cd Path-Planning-A-star-Algorithm/src
```

* Run the program.
```
python main.py
```

* Follow the on screen instructions to provide the input data in order to plan the path of the robot using the A* algorithm.


## Authors

👤 **Ninad Harishchandrakar**

* [GitHub](https://github.com/ninadharish)
* [Email](mailto:ninad.harish@gmail.com)
* [LinkedIn](https://linkedin.com/in/ninadharish)
