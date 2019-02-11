using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Trainers;

namespace ThermopileNeuralNetwork
{
    public class Training
    {
        public enum TrainingAlgorithm
        {
            STOCHASTIC_DUAL_COORDINATE_ASCENT,
            LOGISTIC_REGRESSION,
            NAIVE_BAYES,
            BINARY_STOCHASTIC_DUAL_COORDINATE_ASCENT,
            FAST_TREE,
            STOCHASTIC_GRADIENT_DESCENT
        }

        private readonly IDataView trainData;
        private readonly IDataView evalData;
        private readonly MLContext context;

        public Training(MLContext context)
        {
            this.context = context;
            List<DataControl.Data> tData = new List<DataControl.Data>();
            List<DataControl.Data> eData = new List<DataControl.Data>();

            foreach (var file in DataControl.FILES)
            {
                var tmp = DataControl.LoadTrainingData(file);
                tData.AddRange(tmp.Take(tmp.Count / 2));
                eData.AddRange(tmp.Skip(tmp.Count / 2).Take(tmp.Count / 2));
            }

            trainData = context.CreateStreamingDataView<DataControl.Data>(tData);
            evalData  = context.CreateStreamingDataView<DataControl.Data>(eData);
        }

        public ITransformer TrainModel(TrainingAlgorithm algorithm, DataControl.TrainingOptions options)
        {
            switch (algorithm)
            {
                case TrainingAlgorithm.LOGISTIC_REGRESSION:
                    return LogisticRegression(options);

                case TrainingAlgorithm.NAIVE_BAYES:
                    return NaiveBayes(options);

                case TrainingAlgorithm.BINARY_STOCHASTIC_DUAL_COORDINATE_ASCENT:
                    return BinaryStochasticDualCoordinateAscent(options);

                case TrainingAlgorithm.FAST_TREE:
                    return FastTree(options);

                case TrainingAlgorithm.STOCHASTIC_DUAL_COORDINATE_ASCENT:
                    return StochasticDualCoordinateAscent(options);

                case TrainingAlgorithm.STOCHASTIC_GRADIENT_DESCENT:
                    return StochasticGradientDescent(options);
                default:
                    return null;
            }
        }

        private ITransformer StochasticDualCoordinateAscent(DataControl.TrainingOptions options)
        {
            var pipeline = context.Transforms.Conversion
                .ConvertType(options.LabelColumn, options.LabelColumn, DataKind.R4)
                .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                    labelColumn: options.LabelColumn,
                    featureColumn: options.FeatureColumn,
                    maxIterations: options.MaxIterations));
            Console.WriteLine("========== Training Stochastic Dual Coordinate Ascent Model =========");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        private ITransformer LogisticRegression(DataControl.TrainingOptions options)
        {
            var pipeline = context.Transforms.Conversion
                .ConvertType(options.LabelColumn, options.LabelColumn, DataKind.R4)
                .Append(context.MulticlassClassification.Trainers.LogisticRegression(
                    labelColumn: options.LabelColumn,
                    featureColumn: options.FeatureColumn));

            Console.WriteLine("============== Create and Train Logistic Regression Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        private ITransformer NaiveBayes(DataControl.TrainingOptions options)
        {
            var pipeline = context.Transforms.Conversion
                .ConvertType(options.LabelColumn, options.LabelColumn, DataKind.R4)
                .Append(context.MulticlassClassification.Trainers.NaiveBayes(
                    labelColumn: options.LabelColumn,
                    featureColumn: options.FeatureColumn));

            Console.WriteLine("============== Create and Train Logistic Regression Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        private ITransformer BinaryStochasticDualCoordinateAscent(DataControl.TrainingOptions options)
        {
            var pipeline = context.BinaryClassification.Trainers.StochasticDualCoordinateAscent(
                labelColumn: options.LabelColumn,
                featureColumn: options.FeatureColumn,
                maxIterations: options.MaxIterations
            );

            Console.WriteLine("============== Create and Train Averaged Perceptron Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        private ITransformer FastTree(DataControl.TrainingOptions options)
        {
            var pipeline = context.BinaryClassification.Trainers.FastTree(
                labelColumn: options.LabelColumn,
                featureColumn: options.FeatureColumn,
                learningRate: options.LearningRate
            );

            Console.WriteLine("============== Create and Train Averaged Perceptron Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        private ITransformer StochasticGradientDescent(DataControl.TrainingOptions options)
        {
            var pipeline = context.BinaryClassification.Trainers.StochasticGradientDescent(
                labelColumn: options.LabelColumn,
                featureColumn: options.FeatureColumn,
                maxIterations: options.MaxIterations,
                initLearningRate: options.LearningRate
            );

            Console.WriteLine("============== Create and Train Averaged Perceptron Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        public void Evaluate(ITransformer model, TrainingAlgorithm algorithm, string labelColumn)
        {
            switch (algorithm)
            {
                case TrainingAlgorithm.STOCHASTIC_DUAL_COORDINATE_ASCENT:
                case TrainingAlgorithm.LOGISTIC_REGRESSION:
                case TrainingAlgorithm.NAIVE_BAYES:
                    EvaluateMulticlass(model, labelColumn);
                    break;
                case TrainingAlgorithm.BINARY_STOCHASTIC_DUAL_COORDINATE_ASCENT:
                case TrainingAlgorithm.FAST_TREE:
                case TrainingAlgorithm.STOCHASTIC_GRADIENT_DESCENT:
                    EvaluateBinary(model, labelColumn);
                    break;
            }
        }

        private void EvaluateBinary(ITransformer model, string labelColumn)
        {

            var predictions = model.Transform(evalData);

            var metrics = context.BinaryClassification.EvaluateNonCalibrated(predictions, labelColumn);

            Console.WriteLine();
            Console.WriteLine("Binary Classification Model evaluation for {0}", labelColumn);
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        private void EvaluateMulticlass(ITransformer model, string labelColumn)
        {
            var predictions = model.Transform(evalData);

            var metrics = context.MulticlassClassification.Evaluate(predictions, labelColumn);

            Console.WriteLine();
            Console.WriteLine("Multiclass metrics evaluation for {0}", labelColumn);
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy Macro: {metrics.AccuracyMacro}");
            Console.WriteLine($"Accuracy Micro: {metrics.AccuracyMicro}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");
        }
    }
}