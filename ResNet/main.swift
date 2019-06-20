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

import TensorFlow
import Python
PythonLibrary.useVersion(3)
import Foundation

let deviceName = "/job:localhost/replica:0/task:0/device:XLA_CPU:0"
_RuntimeConfig.useLazyTensor = true
_RuntimeConfig.registerMaterializationCallback { (what: String) in
    if getenv("SWIFT_TENSORFLOW_ENABLE_DEBUG_LOGGING") == nil { return }
    if what != "lazy" { return } 
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

let batchSize = 100

let cifarDataset = loadCIFAR10()
let testBatches = cifarDataset.test.batched(batchSize)

// ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
// PreActivatedResNet18, PreActivatedResNet34
var model = ResNet18(imageSize: 32, classCount: 10) // Use the network sized for CIFAR-10

// the classic ImageNet optimizer setting diverges on CIFAR-10
// let optimizer = SGD(for: model, learningRate: 0.1, momentum: 0.9)
let optimizer = SGD(for: model, learningRate: 0.001)

print("Starting training...")
Context.local.learningPhase = .training

for epoch in 1...1 {
    var trainingLossSum: Float = 0
    var trainingBatchCount = 0
    let trainingShuffled = cifarDataset.training.shuffled(
        sampleCount: 50000, randomSeed: Int64(epoch))
    var step = 0
    for batch in trainingShuffled.batched(batchSize) {
        let start = DispatchTime.now()
        let (labels, images) = (batch.label, batch.data)
        // Materialize the tensors.
        _ = labels._rawTensorHandle
        _ = images._rawTensorHandle
        withDevice(named: deviceName) {
            let (loss, gradients) = valueWithGradient(at: model) { model -> Tensor<Float> in
                let logits = model(images)
                return softmaxCrossEntropy(logits: logits, labels: labels)
            }
            trainingLossSum += loss.scalarized()
            trainingBatchCount += 1
            optimizer.update(&model.allDifferentiableVariables, along: gradients)
        }
        let end = DispatchTime.now()
        let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
        let milliseconds = nanoseconds / 1e6
        print("Epoch: \(epoch) Step: \(step) Time: \(milliseconds) ms")
        step += 1
    }
    var testLossSum: Float = 0
    var testBatchCount = 0
    var correctGuessCount = 0
    var totalGuessCount = 0
    for batch in testBatches {
        let (labels, images) = (batch.label, batch.data)
        // Materialize the tensors.
        _ = labels._rawTensorHandle
        _ = images._rawTensorHandle
        withDevice(named: deviceName) {
            let logits = model(images)
            testLossSum += softmaxCrossEntropy(logits: logits, labels: labels).scalarized()
            testBatchCount += 1
            
            let correctPredictions = logits.argmax(squeezingAxis: 1) .== labels
            correctGuessCount = correctGuessCount +
            Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            totalGuessCount = totalGuessCount + batchSize
        }
    }

    let accuracy = Float(correctGuessCount) / Float(totalGuessCount)
    print("""
          [Epoch \(epoch)] \
          Accuracy: \(correctGuessCount)/\(totalGuessCount) (\(accuracy)) \
          Loss: \(testLossSum / Float(testBatchCount))
          """)
}
