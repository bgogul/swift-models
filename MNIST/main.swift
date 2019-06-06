// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlow

_RuntimeConfig.useLazyTensor = true

// LazyTensorOperation.registerMaterializationCallback { (what: String) in
//   if getenv("SWIFT_TENSORFLOW_ENABLE_DEBUG_LOGGING") == nil { return }
//   print ("----Evaluate called from:")
//   print ("[STACKTRACE-\(what)] start-\(what)")
//   for sym in Thread.callStackSymbols {
//     let components = sym.split(separator: "(")
//     if components.count < 2 {
//       print ("[STACKTRACE-\(what)] <invalid-format>")
//     }
//     let rest = components[1].split(separator: ")")
//     print("[STACKTRACE-\(what)] \(rest[0])")
//   }
//   print("---")
//   fflush(stdout)
// }

var wallTimes: [String: Array<Double>] = ["tffunction": [], "execute":[]]
var startTimes: [Int: DispatchTime] = [0: DispatchTime.now()]
LazyTensor._materializationCallback = { (what: String) in
    if getenv("SWIFT_TENSORFLOW_ENABLE_DEBUG_LOGGING") != nil &&
    (what == "lazy" || what == "Your constant!") {
        print ("----Evaluate called from:")
        print ("[STACKTRACE-\(what)] start-\(what)")
        for sym in Thread.callStackSymbols {
            let components = sym.split(separator: "(")
            if components.count < 2 {
                print ("[STACKTRACE-\(what)] <invalid-format>")
            }
            let rest = components[1].split(separator: ")")
            print("[STACKTRACE-\(what)] \(rest[0])")
        }
        print("---")
        fflush(stdout)
    }
  let startTime = startTimes[0]!
  if (what != "eager") && (what != "lazy") {
    let end = DispatchTime.now()
    let nanoseconds = Double(end.uptimeNanoseconds - startTime.uptimeNanoseconds)
    let milliseconds = nanoseconds / 1e6
    if wallTimes[what] != nil {
      wallTimes[what]!.append(milliseconds)
    } else {
      wallTimes[what] = [milliseconds]
    }
  }
  startTimes[0] = DispatchTime.now()
}


/// Reads a file into an array of bytes.
func readFile(_ path: String) -> [UInt8] {
    let possibleFolders  = [".", "MNIST"]
    for folder in possibleFolders {
        let parent = URL(fileURLWithPath: folder)
        let filePath = parent.appendingPathComponent(path)
        guard FileManager.default.fileExists(atPath: filePath.path) else {
            continue
        }
        let data = try! Data(contentsOf: filePath, options: [])
        return [UInt8](data)
    }
    print("File not found: \(path)")
    exit(-1)
}

/// Reads MNIST images and labels from specified file paths.
func readMNIST(imagesFile: String, labelsFile: String) -> (images: Tensor<Float>,
                                                           labels: Tensor<Int32>) {
    print("Reading data.")
    let images = readFile(imagesFile).dropFirst(16).map(Float.init)
    let labels = readFile(labelsFile).dropFirst(8).map(Int32.init)
    let rowCount = labels.count
    let imageHeight = 28, imageWidth = 28

    print("Constructing data tensors.")
    return (
        images: Tensor(shape: [rowCount, 1, imageHeight, imageWidth], scalars: images)
            .transposed(withPermutations: [0, 2, 3, 1]) / 255, // NHWC
        labels: Tensor(labels)
    )
}

/// A classifier.
struct Classifier: Layer {
    typealias Input = Tensor<Float>
    typealias Output = Tensor<Float>

    var conv1a = Conv2D<Float>(filterShape: (3, 3, 1, 32), activation: relu)
    var conv1b = Conv2D<Float>(filterShape: (3, 3, 32, 64), activation: relu)
    var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

    var dropout1a = Dropout<Float>(probability: 0.25)
    var flatten = Flatten<Float>()
    var layer1a = Dense<Float>(inputSize: 9216, outputSize: 128, activation: relu)
    var dropout1b = Dropout<Float>(probability: 0.5)
    var layer1b = Dense<Float>(inputSize: 128, outputSize: 10, activation: softmax)

    @differentiable
    func callAsFunction(_ input: Input) -> Output {
        let convolved = input.sequenced(through: conv1a, conv1b, pool1)
        return convolved.sequenced(through: dropout1a, flatten, layer1a, dropout1b, layer1b)
    }
}

// setenv("SWIFT_TENSORFLOW_SERVER_ADDRESS", "grpc://localhost:51000", 0)
// withDevice(named: "/job:localhost/replica:0/task:1/device:TPU:0") {
// withDevice(named: "/job:localhost/replica:0/task:0/device:XLA_CPU:0") {
// withDevice(named: "/job:localhost/replica:0/task:0/device:CPU:0") {

let epochCount = 1
let batchSize = 128

func minibatch<Scalar>(in x: Tensor<Scalar>, at index: Int) -> Tensor<Scalar> {
    let start = index * batchSize
    return x[start..<start+batchSize]
}

let (trainImages, trainNumericLabels) = readMNIST(imagesFile: "train-images-idx3-ubyte",
                                                  labelsFile: "train-labels-idx1-ubyte")
let trainLabels = Tensor<Float>(oneHotAtIndices: trainNumericLabels, depth: 10)

let (testImages, testNumericLabels) = readMNIST(imagesFile: "t10k-images-idx3-ubyte",
                                                labelsFile: "t10k-labels-idx1-ubyte")
let testLabels = Tensor<Float>(oneHotAtIndices: testNumericLabels, depth: 10)

var classifier = Classifier()

let optimizer = Adam(for: classifier)

print("Beginning training...")

// struct Statistics {
//     var correctGuessCount: Tensor<Int32> = Tensor<Int32>(0)
//     var totalGuessCount: Tensor<Int32> = Tensor<Int32>(0)
//     var totalLoss: Tensor<Float> = Tensor<Float>(0.0)
// }
struct Statistics {
    var correctGuessCount: Int = 0
    var totalGuessCount: Int = 0
    var totalLoss: Float = 0
}


var times = [Double]()
let executions = wallTimes["execute"]!.count - 0
print ("Executions beforels loop: \(executions)")
var tfeExecutes = executions

// The training loop.
for epoch in 1...epochCount {
    var trainStats = Statistics()
    var testStats = Statistics()
    Context.local.learningPhase = .training
    let trainImagesDS = Dataset<Tensor<Float>>(elements: trainImages).batched(batchSize)
    let trainLabelsDS = Dataset<Tensor<Int32>>(elements: trainNumericLabels).batched(batchSize)
    var i = 0
    for (x, y) in zip(trainImagesDS, trainLabelsDS) {
        /// Materialize
        // let yc = y._rawTensorHandle
        // let xc = x._rawTensorHandle
    // for i in 0 ..< Int(trainLabels.shape[0]) / batchSize {
        // for i in 0 ..< 5 {
        if (i > 5) {
            i += 1
            break
        }
        let start = DispatchTime.now()
        // let x = minibatch(in: trainImages, at: i)
        // let y = minibatch(in: trainNumericLabels, at: i)
        // Compute the gradient with respect to the model.
        let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
            let ŷ = classifier(x)
            let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y
            trainStats.correctGuessCount += Int(
                Tensor<Int32>(correctPredictions).sum().scalarized())
            trainStats.totalGuessCount += batchSize
            let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
            trainStats.totalLoss += loss.scalarized()
            return loss
        }
        // Update the model's differentiable variables along the gradient vector.
        optimizer.update(&classifier.allDifferentiableVariables, along: 𝛁model)
        let end = DispatchTime.now()
        let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
        let milliseconds = nanoseconds / 1e6
        times.append(milliseconds)
        let executions = wallTimes["execute"]!.count - tfeExecutes
        tfeExecutes = wallTimes["execute"]!.count
        print("Epoch: \(epoch) Step: \(i) Time: \(milliseconds) ms Executions: \(executions) ")
    }

    Context.local.learningPhase = .inference
    for i in 0 ..< Int(testLabels.shape[0]) / batchSize {
        let x = minibatch(in: testImages, at: i)
        let y = minibatch(in: testNumericLabels, at: i)
        // Compute loss on test set
        let ŷ = classifier(x)
        let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y
        testStats.correctGuessCount += Int(
            Tensor<Int32>(correctPredictions).sum().scalarized())
        testStats.totalGuessCount += batchSize
        let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
        testStats.totalLoss += loss.scalarized()
    }

    let trainAccuracy = Float(trainStats.correctGuessCount) / Float(trainStats.totalGuessCount)
    let testAccuracy = Float(testStats.correctGuessCount) / Float(testStats.totalGuessCount)
    print("""
          [Epoch \(epoch)] \
          Training Loss: \(trainStats.totalLoss), \
          Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalGuessCount) \ 
          (\(trainAccuracy)), \
          Test Loss: \(testStats.totalLoss), \
          Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalGuessCount) \
          (\(testAccuracy))
        """)
    for (what, times) in wallTimes {
        if times.count == 0 { continue }
        print("\(what) events: \(times.count) average: \(times.reduce(0.0, +)/Double(times.count)) ms, sum: \(times.reduce(times[0], +)), " +
            "min: \(times.reduce(times[0], min)) ms,   " +
            "max: \(times.reduce(times[0], max)) ms")
    }
    print("step events: \(times.count) average: \(times.reduce(0.0, +)/Double(times.count)) ms, sum: \(times.reduce(times[0], +)) " +
        "min: \(times.reduce(times[0], min)) ms,   " +
        "max: \(times.reduce(times[0], max)) ms")

    // if (true) {
    //     let x = wallTimes["tffunction"]!
    //     let y = wallTimes["execute"]!
    //     var c = 0
    //     for (a, b) in zip(x,y) {
    //         print("\(c): \(a), \(b)")
    //         c += 1
    //     }
    // }
}
// } // withDevice
