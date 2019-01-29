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
        public static ITransformer TrainContainsPeople(MLContext context)
        {
            List<DataControl.Data> data = new List<DataControl.Data>();
            
            
            foreach (var file in Constants.FILES)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Take(tmp.Count/2));
            }

            var trainData = context.CreateStreamingDataView<DataControl.Data>(data);

            var pipeline = context.BinaryClassification.Trainers.StochasticGradientDescent(
                labelColumn: "ContainsPeople",
                featureColumn: "Features");

            Console.WriteLine("============== Create and Train ContainsPeople Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        public static ITransformer TrainNumberOfPeople(MLContext context)
        {
            List<DataControl.Data> data = new List<DataControl.Data>();
            
            
            foreach (var file in Constants.FILES)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Take(tmp.Count/2));
            }

            var trainData = context.CreateStreamingDataView<DataControl.Data>(data);

            var pipeline = context.Transforms.Conversion.ConvertType("NumberOfPeople","NumberOfPeople",DataKind.R4)
                .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                labelColumn: "NumberOfPeople",
                featureColumn: "Features"));

            Console.WriteLine("============== Create and Train NumberOfPeople Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }

        public static ITransformer TrainDistance(MLContext context)
        {
            List<DataControl.Data> data = new List<DataControl.Data>();
            
            
            foreach (var file in Constants.FILES)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Take(tmp.Count/2));
            }
            
            var trainData = context.CreateStreamingDataView<DataControl.Data>(data);

            var pipeline = context.Transforms.Conversion.ConvertType("Distance","Distance",DataKind.R4)
                .Append(context.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                            labelColumn: "Distance",
                            featureColumn: "Features"));

            Console.WriteLine("============== Create and Train Distance Model ==============");
            var model = pipeline.Fit(trainData);
            Console.WriteLine("================= Finished Training ================");
            Console.WriteLine();
            return model;
        }
        
        public static void EvaluateContainsPeople(MLContext context, ITransformer model, List<FileData> fileData)
        {
            List<DataControl.Data> data = new List<DataControl.Data>();

            foreach (var file in fileData)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Skip(tmp.Count/2).Take(tmp.Count/2));
            }
            var trainData = context.CreateStreamingDataView<DataControl.Data>(data);

            var predictions = model.Transform(trainData);

            var metrics = context.BinaryClassification.EvaluateNonCalibrated(predictions, "ContainsPeople");
            
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public static void EvaluateNumberOfPeople(MLContext context, ITransformer model, List<FileData> fileData)
        {
            List<DataControl.Data> data = new List<DataControl.Data>();

            foreach (var file in fileData)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Skip(tmp.Count/2).Take(tmp.Count/2));
            }
            var trainData = context.CreateStreamingDataView<DataControl.Data>(data);

            var predictions = model.Transform(trainData);

            var metrics = context.MulticlassClassification.Evaluate(predictions, "NumberOfPeople");

            Console.WriteLine();
            Console.WriteLine("NumberOfPeople metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy Macro: {metrics.AccuracyMacro}");
            Console.WriteLine($"Accuracy Micro: {metrics.AccuracyMicro}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");
        }
        
        public static void EvaluateDistance(MLContext context, ITransformer model, List<FileData> fileData)
        {
            List<DataControl.Data> data = new List<DataControl.Data>();

            foreach (var file in fileData)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Skip(tmp.Count/2).Take(tmp.Count/2));
            }
            var trainData = context.CreateStreamingDataView<DataControl.Data>(data);

            var predictions = model.Transform(trainData);

            var metrics = context.MulticlassClassification.Evaluate(predictions, "Distance");

            Console.WriteLine();
            Console.WriteLine("Distance metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy Macro: {metrics.AccuracyMacro}");
            Console.WriteLine($"Accuracy Micro: {metrics.AccuracyMicro}");
            Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");
        }
    }
}