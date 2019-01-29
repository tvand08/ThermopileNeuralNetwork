using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;

namespace ThermopileNeuralNetwork
{
    public class Constants
    {

        public static readonly string MODEL_PATH = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

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