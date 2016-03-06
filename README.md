#A Recommendation System for Symmetric Tensor Data 
#####Sol Vitkin and Rishi Sharma 
#####Last Update: February 14, 2016 


OBJECTIVE: For a group of objects, we are looking at and predicting how well an interaction of those objects perform along some metric or over a series of different assessments. We are specifically looking at a 3-rank a x a x N tensor, were each a_ijk refers to object ai interacting with aj in assessment k. We use our tensor factoization alogirthm that finds weights with a list of pattern matrices to represent each slice of our tensor. We use a coordinate descent algorithm to converge on a solution. 

FILES: tensorfactorization.py

RUNTIME INSTRUCTIONS: File runs in Python 2.7 and requires pandas, numpy, random, math, time, and argparse packages. When running file from command line, additional argument required(related to data generation): 'normal' for data to be created from Normal distribution or 'uniform' for data to be created from Uniform distribution.

The relevant results are printed. 


Enjoy!
