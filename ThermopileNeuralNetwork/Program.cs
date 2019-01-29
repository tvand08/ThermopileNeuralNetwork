using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Normalizers;

namespace ThermopileNeuralNetwork
{
   
    
    
    class Program
    {
        
        public Program(string[] args)
        {
            var context = new MLContext();

            var containsPeopleModel = Training.TrainContainsPeople(context);
            var numberOfPeopleModel = Training.TrainNumberOfPeople(context);
            var distanceModel = Training.TrainDistance(context);
             
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");

            Training.EvaluateContainsPeople(context, containsPeopleModel, Constants.FILES);
            Training.EvaluateNumberOfPeople(context, numberOfPeopleModel, Constants.FILES);
            Training.EvaluateDistance(context, distanceModel, Constants.FILES);
           
            Console.WriteLine("=============== End of model evaluation ===============");
            
            Console.WriteLine("=============== Consuming Model ===============");
            Console.WriteLine();
            
            var predictor = new Predictor(context, containsPeopleModel, numberOfPeopleModel, distanceModel);

            List<DataControl.Data> data = new List<DataControl.Data>();
            foreach (var file in Constants.FILES)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Skip(tmp.Count/2).Take(tmp.Count/2));
            }

            data = RandomPermutation(data);
            foreach (var d in data)
            {
                predictor.Predict(d);
            }
            Console.WriteLine("=========== Finished Consuming Model =============");

        }

        private static void SaveModelAsZip(MLContext context, ITransformer model)
        {
            using (var fileStream = new FileStream(Constants.MODEL_PATH, FileMode.Create, FileAccess.Write, FileShare.Write))
                context.Model.Save(model, fileStream);
            
            Console.WriteLine("The model is saved to {0}", Constants.MODEL_PATH);
        }
        
        static Random random = new Random();

        public static List<T> RandomPermutation<T>(List<T> sequence)
        {
            T[] retArray = sequence.ToArray();

            for (int i = 0; i < retArray.Length - 1; i += 1)
            {
                int swapIndex = random.Next(i, retArray.Length);
                if (swapIndex != i) {
                    T temp = retArray[i];
                    retArray[i] = retArray[swapIndex];
                    retArray[swapIndex] = temp;
                }
            }

            return retArray.ToList();
        }
        
        static void Main(string[] args)
        {
            new Program(args);
        }
    }
}