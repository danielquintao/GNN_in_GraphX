
// =============================================================================
// 1- load and parse dataset
// For the moment, create a fake graph with fake features and labels
// =============================================================================

import org.apache.spark.graphx.{Graph, VertexId, PartitionStrategy, VertexRDD}
import org.apache.spark.graphx.util.GraphGenerators
import org.apache.spark.rdd.RDD
import scala.util.Random
import breeze.linalg._
import breeze.stats.distributions.Gaussian
import scala.math.{log, exp, sqrt}

val rand = scala.util.Random
val nIn = 2+1  // number of features INCLUDING BIAS
val nHid = 3  // number of neurons in hidden layer
val nOut = 1  // dimension of label

// =============================================================================
// 2- train a "common" NN before doing any GNN
// =============================================================================
// training hyper-parameters
val lr = 0.01 // learning rate

// parameters for model
// Kaiming He initialization for layer with ReLU:
val gaussianSampler = Gaussian(0, sqrt(2.0 / nIn))
var w1 = DenseMatrix.tabulate(nIn, nHid){(i,j) => gaussianSampler.sample(1)(0)}
w1(-1, ::) := 0.0 // bias set to 0
// Xavier Glorot initialization for layer with sigmoid:
var w2 = (DenseMatrix.rand(nHid + 1, nOut) - 0.5) / sqrt(nHid + 1)
w2(-1, ::) := 0.0 // bias set to 0

class NN(var useBCE: Boolean = true) {//! DEPRECATED

  // values we should cache for the backpropagation
  var in: DenseMatrix[Double] = _
  var z1: DenseMatrix[Double] = _
  var a1: DenseMatrix[Double] = _
  var z2: DenseMatrix[Double] = _
  var a2: DenseMatrix[Double] = _

  def forward(input: DenseMatrix[Double], w1: DenseMatrix[Double], w2: DenseMatrix[Double], debug: Boolean = false): DenseMatrix[Double] = {
    in = input
    z1 = input * w1  // matrix product
    if (debug) println("z1=\n" + z1)
    a1 = z1.copy
    a1(z1 <:< DenseMatrix.zeros[Double](z1.rows, z1.cols)) := 0.0 // ReLU
    if (debug) println("a1=\n" + a1)
    z2 = DenseMatrix.horzcat(a1, DenseMatrix.ones[Double](a1.rows, 1)) * w2  // matrix product; note we added the bias dim
    if (debug) println("z2=\n" + z2)
    a2 = 1.0 /:/ (1.0 +:+ (-z2).map(exp)) // sigmoid
    if (debug) println("a2=\n" + a2)
    a2
  }

  def backward(gt: DenseMatrix[Double], w1: DenseMatrix[Double], w2: DenseMatrix[Double],  debug: Boolean = false): Option[(DenseMatrix[Double], DenseMatrix[Double])] = {
    // gt is the ground-truth (true label)
    if (useBCE) {
      // compute loss
      val loss = -gt *:* a2.map(x => scala.math.log(x + 1E-8)) - (1.0-gt) *:* (1.0-a2).map(x => scala.math.log(x + 1E-8))
      if(debug) {println(loss)}
      
      // compute gradients
      val gradLoss = a2.copy
      gradLoss((a2 :== 0.0) & (gt :== 0.0)) := 0.0 // avoid numerical error
      gradLoss((a2 :== 0.0) & (gt :== 1.0)) := -99999.0 // avoid numerical error
      gradLoss((a2 :== 1.0) & (gt :== 1.0)) := 0.0 // avoid numerical error
      gradLoss((a2 :== 1.0) & (gt :== 0.0)) := 99999.0 // avoid numerical error
      val a2_comp = 1.0 - a2  // doing (1.0 - x)(cond) for DenseMatrix x and cond (boolean) yields error, so we define x_comp and use it
      val gt_comp = 1.0 - gt
      val cond = !(a2 :== 0.0) & !(a2 :== 1.0)
      gradLoss(cond) := - gt(cond) /:/ a2(cond) + gt_comp(cond) /:/ a2_comp(cond)
      if(debug) {println("gradLoss=\n" + gradLoss)}

      val da2dz2 = a2 *:* (1.0 - a2) // derivative of sigmoid
      if(debug) {println("da2/dz2=\n" + da2dz2)}
      val delta2 = gradLoss *:* da2dz2
      if(debug) {println("delta2=\n" + delta2)}

      val da1dz1 = a1.copy
      da1dz1(a1 >:> 0.0) := 1.0 // derivative of Relu is 1 in (strictly) pos entries, zero everywhere else (NOTE that a1 has no neg entry)
      if(debug) {println("da1/dz1=\n" + da1dz1)}
      val delta1 = (delta2 * w2(0 to -2, ::).t) *:* da1dz1  // w2(:-1)=dz2/d(a1(:-1)), delta2=dLoss/dz2, so we get delta1=dLoss/dz1
      if(debug) {println("delta1=\n" + delta1)}

      val dw2 = DenseMatrix.horzcat(a1, DenseMatrix.ones[Double](a1.rows, 1)).t * delta2
      if(debug) {println("dw2=\n" + dw2)}

      val dw1 = in.t * delta1
      if(debug) {println("dw1=\n" + dw1)}

      // return gradients
      Some(dw1, dw2)
    } else {
      None
    }
  }
}
val nn = new NN()
val x = DenseMatrix.rand(5, nIn)  // fake input
val yt = DenseMatrix.ones[Double](5, nOut)  // fake true labels
yt(DenseMatrix.rand(5, nOut) <:< 0.5) := 0.0
nn.forward(x, w1, w2, true)
nn.backward(yt, w1, w2, true) // only works after the forward

// =============================================================================
// 3- Generation of our super graph with data nodes and parameter nodes
// =============================================================================

class LearningData(var features: DenseVector[Double], var label: Double)

// some definitions //? do we need it?
trait ParentVertexProperty //! declaring ParentVertexProperty as a class does not work when spark tries to deserialize PArameterVertexProperty
case class DataVertexProperty(
  val data: LearningData // or val data: (DenseVector[Double], Double)
) extends ParentVertexProperty
case class ParameterVertexProperty(
  var w1: Option[DenseMatrix[Double]] = None,
  var w2: Option[DenseMatrix[Double]] = None,
  // val lr: Double, //? do we need it?
  // Variables for backpropagation:
  var in: Option[DenseMatrix[Double]] = None,
  var z1: Option[DenseMatrix[Double]] = None,
  var a1: Option[DenseMatrix[Double]] = None,
  var z2: Option[DenseMatrix[Double]] = None,
  var a2: Option[DenseMatrix[Double]] = None
) extends ParentVertexProperty


// -----------------------------------------------------------------------------
// 3.1 create our single graph with both Data and Parameter vertices
// -----------------------------------------------------------------------------

// 3.1.1 - Create data vertices, annotated with features and labels
val dataGraph: Graph[LearningData, Int] =
  GraphGenerators.logNormalGraph(sc, numVertices = 100).mapVertices( // obs: may have "recursive edges""
    (_, _) => {
      var data = new
        LearningData(
          DenseVector.rand(nIn),
          randomInt()  // 0 or 1
        )
      data.features(-1) = 1  // bias 
      data
    }
  )
  
// 3.1.2 - Make the dataGraph undirected (in GraphX, this means duplicating+reversing all edges)
var graph = Graph.apply(
  vertices=dataGraph.vertices,
  edges=dataGraph.edges.union(dataGraph.edges.reverse)
)
// Merge edges pointing in the same direction, specially if we already received some pairs A-B,B-A:
//? (are we supposed to do this "deduplication" step or is the dataset a multigraph?)
graph = graph.partitionBy(PartitionStrategy.CanonicalRandomVertexCut) // NOTE that groupEdges requires
//                                                    a previous partitioning step to co-localize edges
graph = graph.groupEdges{case (ed1, ed2) => ed1} // edges have no relevant annotation in our scenario,
//                                                  so we simply take the first one 


// 3.1.3 - Add parameter vertices where each has "batch-size" data->parameter edges.
// A further improvement on the Routing Table (where GraphX says, for each vertex, in which partitions
// we find its edges) is to use a constraint on the IDs of those data vertices that point toward the
// the parameter vertex, e.g. that they are all congruent mod "nb partitions".
val nPartitions = graph.edges.getNumPartitions
val mixingPrime: VertexId = 1125899906842597L //* MUST BE THE SAME USED BY PartitionStrategy, c.f. https://github.com/apache/spark/blob/v3.3.1/graphx/src/main/scala/org/apache/spark/graphx/PartitionStrategy.scala
val batchSize = 4 //TODO change to 512 with big dataset
// 3.1.3.1 - The idea here is to have an RDD where each entry is a list of AT MOST batchSize vertices FROM THE SAME PARTITION
// plus an "ID" computed as k*nPart + i, where 0 <= i < nPartitions is the partition where the vertices composing the
// list are co-located and k is the counter of this list of vertices within the partition
// (we will use this "ID" below to build the parameter Nodes and know which vertices point to it)
var groupedVertices =  graph.vertices.map(x => (x._1 * mixingPrime % nPartitions, x._1)).
                            groupByKey().
                            map(x => (x._1, x._2.grouped(batchSize).toList)).
                            flatMap(x => {
                              for ((innerList, k) <- x._2.zipWithIndex) yield (k * nPartitions + x._1, innerList)
                              }
                            )
// 3.1.3.2 - create the parameter vertices.
val maxDataVertexId = graph.vertices.map(x => x._1).max
// smallest multiple of nPartitions that doesn't coincide with any data vertex id:
val parameterVertexOffset = nPartitions * (maxDataVertexId / nPartitions + 1)
// create the parameter vertices properly said
val vrdd: RDD[(VertexId, ParameterVertexProperty)] = groupedVertices.map(x => {
    // Kaiming He initialization for layer with ReLU:
    val gaussianSampler = Gaussian(0, sqrt(2.0 / nIn))
    var w1 = DenseMatrix.tabulate(nIn, nHid){(i,j) => gaussianSampler.sample(1)(0)}
    w1(-1, ::) := 0.0 // bias set to 0
    // Xavier Glorot initialization for layer with sigmoid:
    var w2 = (DenseMatrix.rand(nHid + 1, nOut) - 0.5) / sqrt(nHid + 1)
    w2(-1, ::) := 0.0 // bias set to 0
    val data = ParameterVertexProperty(Some(w1), Some(w2))
    (x._1 + parameterVertexOffset, data)
  }
)
val parameterVertices: VertexRDD[ParameterVertexProperty] = VertexRDD(vrdd)  // convert to VertexRDD format
// 3.1.3.3 - build a new graph with both parameter and data nodes
// TODO <- I'M HERE

// 3.1.4 - Partition the graph so that all edges point to the same node are together
// (partition by dst vertex id)

//! ATTENTION it seems that GraphSage was made for undirected graphs, so we need to replicate all
//! nodes in our **data** graph