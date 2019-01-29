using System;
using System.Collections.Generic;
using System.Linq;
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
    }
}