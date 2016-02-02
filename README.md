A Recommendation System for Symmetric Tensor Data
Sol Vitkin and Rishi Sharma 
Last Update: December 20, 2015


OBJECTIVE: For a group of objects, we are looking at and predicting how well an interaction of those objects perform along some metric or over a series of different assessments. We are specifically looking at a 3-rank a x a x N tensor, were each a_ijk refers to object ai interacting with aj in assessment k. We use our tensor factoization alogirthm that finds weights with a list of pattern matrices to represent each slice of our tensor. We use a coordinate descent algorithm to converge on a solution. 

FILES: tensorfactorization.py

RUNTIME INSTRUCTIONS: Within the file at the bottom you will see  that we create data using the create_data(text). Text can either be 'uniform' where it will generate data through a uniform distribution within our projected range, or it can be 'normal' where it will normally generate the data. 

The relevant results are printed. 


Enjoy!
