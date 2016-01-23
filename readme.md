# CloudToGrid

This repo shows how to map one point cloud to another; for example, creating a 2d grid embedding of a t-SNE point cloud.

This repo uses [pyLAPJV](https://x.st/code.html#pyLAPJV) wrapper for Jonker's implementation of the [Jonker-Volgenant algorithm](http://link.springer.com/article/10.1007%2FBF02278710) as well as [hungarian](https://github.com/hrldcpr/hungarian), which are both solutions to the "[assignment problem](https://en.wikipedia.org/wiki/Assignment_problem)", a combinatorial optimization problem that asks: how do you assign workers to tasks in a way that minimizes the overall cost? For CloudToGrid, the "task" is point correspondence, and the cost of each correspondence is the squared distance between the original and destination point. The Jonker-Volgenant algorithm is a less tolerant but more efficient solution than the traditional [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm).

The Jonker-Volgenant algorithm implementation used here can fail when the difference between two costs is very small, and both algorithms can fail when the costs are all small.

## Installation and Usage

First, install numpy, scipy, sklearn, matplotlib, [pyLAPJV](https://x.st/code.html#pyLAPJV) and [hungarian](https://github.com/hrldcpr/hungarian):

```
$ pip install numpy scipy sklearn matplotlib
$ pip install git+https://github.com/hrldcpr/pyLAPJV.git
$ pip install git+https://github.com/hrldcpr/hungarian.git
```

Then clone this repository and run the example:

```
$ git clone https://github.com/kylemcdonald/CloudToGrid.git
$ cd CloudToGrid
$ jupyter notebook
```

## Acknowledgement

This repository is inspired by Mario Klingemann's [Raster Fairy](https://github.com/Quasimondo/RasterFairy) work.