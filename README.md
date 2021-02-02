# pp2

This project provides a more robust and stable version of the Principal Path algorithm (1).

1. 'Finding Prinicpal Paths in Data Space', M.J.Ferrarotti, W.Rocchia, S.Decherchi

## Downloads files (mandatory)

```
git clone https://github.com/erikagardini/pp2.git
```

## Hot to use this code

You can install python requirements with

```
pip3 install -r requirements.txt
```

## Run the experiments with the new proposed method

You can compute the Principal Path with the new proposed method as follow:

```
cd new_principalpath
python3 "name_of_the_experiment".py
```
where the parameter **name_of_the_experiment** can assume the following value:
- "2d_experiment": to reproduce the experiments with the 2d trivial data sets
- "face_olivetti_experiment": to reproduce the experiment with the olivetti data set
- "mnist_experiment": to reproduce the experiment with the mnist data set
  
The code produces the following output when the 2d experiment is run:
- the Dijkstra shortest path from the starting point to the ending point
- the adjasted path with points equally distributed along the shortest path
- the image of the path for different values of s
- the value of the evidence for each model, it allows to perform the model selection step

The code produces the following output when an high dimensional experiment is run:
- the starting figure
- the ending figure
- the Dijkstra shortest path from the starting point to the ending point
- the adjasted path with points equally distributed along the shortest path
- the resulting path for different values of s
- the nearest figures to each path waypoints for each model

## Run the experiments with the original version of the method

You can compute the Principal Path with the original method as follow:

```
cd original_principalpath
python3 "name_of_the_experiment".py
```
where the parameter **name_of_the_experiment** can assume the following value:
- "2d_experiment": to reproduce the experiments with the 2d trivial data sets
- "face_olivetti_experiment": to reproduce the experiment with the olivetti data set
- "mnist_experiment": to reproduce the experiment with the mnist data set
  
The code produces the following output when the 2d experiment is run:
- the filtered data (if the prefiltering procedure is performed)
- the waypoints initialization
- the image of the path for different values of s
- the value of the evidence for each model, it allows to perform the model selection step

The code produces the following output when an high dimensional experiment is run:
- the starting figure
- the ending figure
- the resulting path for different values of s
