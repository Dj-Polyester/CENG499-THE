## IMPORTANT

* In `Distance.py`, instead of the expression for the Cosine similarity function as stated in the assignment text, negated version of it is used. The reason for this is explained in the report for part 1.

* In part 2, instead of a separate K-Means++ class in `KMeansPlusPlus.py`, 
an initializer parameter in `KMeans` class is used. The default 
argument for `initializer` is `KMeans.KMeansMinusMinus` static function, 
which is actually a purely random initialization method. There is also 
`KMeans.KMeansPlusPlus`, the initialization method used in the KMeans++ 
algorithm. The details are explained in the report for part 2.

* Note that part 2 is also using `Distance.py`