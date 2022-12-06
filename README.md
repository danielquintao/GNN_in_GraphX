# GraphSAGE in GraphX

In order to run the code in ``gnn.scala`` in a Spark Shell

1. download ppi data ("preprocessed" version) from https://snap.stanford.edu/graphsage/

2. Run ``convert_features_file_ppi.py`` to convert the file ``ppi-features.npy`` into a scala-friendly format

3. fix the path to the data in ``gnn.scala``

4. open the spark-shell with enough RAM (e.g. ``cd /spark-3.3.1-bin-hadoop3``
   and then ``./bin/spark-shell --driver-memory 8g``), paste the content of ``gnn.scala`` and have fun
   
Feel free to adapt the code to a standalone spark app