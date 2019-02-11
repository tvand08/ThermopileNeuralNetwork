using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CommandLine;
using Microsoft.ML.Data;

namespace ThermopileNeuralNetwork
{
    public class DataControl
    {
        public class Data
        {
            [ColumnName("Features")] [VectorType(64)]
            public float[] Features;
        
            [ColumnName("ContainsPeople")] 
            public bool ContainsPeople;

            [ColumnName("NumberOfPeople")] 
            public int NumberOfPeople;

            [ColumnName("Distance")] 
            public int Distance;
        }

        public class PredictedContainsPeople
        {
            [ColumnName("PredictedLabel")] 
            public bool ContainsPeople { get; set; }
        }

        public class PredictedNumberOfPeople
        {
            [ColumnName("PredictedLabel")]
            public UInt32 NumberOfPeople { get; set; }
        }
             
        public class PredictedDistance
        {
            [ColumnName("PredictedLabel")] 
            public UInt32 Distance { get; set; }
        }

        public class PredictionResult
        {
            public bool ActualContainsPeople { get; set; }
            public int ActualNumberOfPeople { get; set; }
            public int ActualDistance { get; set; }
            
            public bool PredictionContainsPeople { get; set; }
            public uint PredictionNumberOfPeople { get; set; }
            public uint PredictionDistance { get; set; }
        }

        public class TrainingOptions
        {
            public string FeatureColumn { get; set; }
            public string LabelColumn { get; set; }
            public int MaxIterations { get; set; } = 20;
            public float LearningRate { get; set; } = 0.2f;
        }

        public class EvaluationOptions
        {
            public string LabelColumn { get; set; }
        }
        
        public static List<Data> LoadTrainingData(FileData file)
        {
            var data = new List<Data>();
            
            string[] lines = System.IO.File.ReadAllLines(file.filepath);

            for (int i = 1; i < lines.Length; i++)
            {
                var datum = lines[i].Split(null);
                var row = new Data();
                
                row.ContainsPeople = file.containsPeople;
                row.NumberOfPeople = file.numberOfPeople;
                row.Distance = file.distance;
                
                row.Features = datum.Skip(1).Take(64).Select(k => float.Parse(k)).ToArray();
                if (row.Features.Length == 0) continue;
                data.Add(row);
            }

            return data;   
        }
        
        public static List<FileData> FILES = new List<FileData>()
        {
            new FileData()
            {
                filepath = Path.Combine(Environment.CurrentDirectory,"data","control","1p3ft"),
                containsPeople = true,
                numberOfPeople = 1,
                distance = 3
            },
            new FileData()
            {
                filepath = Path.Combine(Environment.CurrentDirectory,"data","control","nothing"),
                containsPeople = false,
                numberOfPeople = 0,
                distance = 0 
            },
            new FileData()
            {
                filepath = Path.Combine(Environment.CurrentDirectory,"data","control","1p3ft"),
                containsPeople = true,
                numberOfPeople = 1,
                distance = 3
            },
            new FileData()
            {
                filepath = Path.Combine(Environment.CurrentDirectory,"data","distance","1p1ft"),
                containsPeople = true,
                numberOfPeople = 1,
                distance = 1
            },
            new FileData()
            {
                filepath = Path.Combine(Environment.CurrentDirectory,"data","distance","1p6ft"),
                containsPeople = true,
                numberOfPeople = 1,
                distance = 6
            },
            new FileData()
            {
                filepath = Path.Combine(Environment.CurrentDirectory,"data","qty","2p3ft"),
                containsPeople = true,
                numberOfPeople = 2,
                distance = 3
            },
            new FileData()
            {
                filepath = Path.Combine(Environment.CurrentDirectory,"data","qty","3p3ft"),
                containsPeople = true,
                numberOfPeople = 3,
                distance = 3
            }
        };

        
    }

    public class FileData
    {
        public string filepath { get; set; }

        public bool containsPeople { get; set; }

        public int numberOfPeople { get; set; }

        public int distance { get; set; }
    }
}