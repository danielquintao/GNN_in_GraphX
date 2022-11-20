
// =============================================================================
// 1- load and parse dataset
// For the moment, create a fake graph with fake features and labels
// =============================================================================

import org.apache.spark.graphx.{Graph, VertexId}
import org.apache.spark.graphx.util.GraphGenerators
import scala.util.Random
import breeze.linalg._
import breeze.stats.distributions.Gaussian
import scala.math.{log, exp, sqrt}

class LearningData(var features: DenseVector[Double], var label: Double)
val rand = scala.util.Random
val nIn = 2+1  // number of features INCLUDING BIAS
val nHid = 3  // number of neurons in hidden layer
val nOut = 1  // dimension of label

// graph of train samples, annotated with features and labels
val graph: Graph[LearningData, Int] =
  GraphGenerators.logNormalGraph(sc, numVertices = 100).mapVertices(
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

class NN(var useBCE: Boolean = true) {

  // values we should cache for the backpropagation
  var in: DenseMatrix[Double] = _
  var z1: DenseMatrix[Double] = _
  var a1: DenseMatrix[Double] = _
  var z2: DenseMatrix[Double] = _
  var a2: DenseMatrix[Double] = _

  def forward(input: DenseMatrix[Double], debug: Boolean = false): DenseMatrix[Double] = {
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

  def backward(gt: DenseMatrix[Double], debug: Boolean = false): Option[(DenseMatrix[Double], DenseMatrix[Double])] = {
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
nn.forward(x, true)
nn.backward(yt, true) // only works after the forward