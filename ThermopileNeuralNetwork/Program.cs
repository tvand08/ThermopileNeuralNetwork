using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CommandLine;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Normalizers;

namespace ThermopileNeuralNetwork
{
   
    
    
    class Program
    {
#region Variables
        private const Training.TrainingAlgorithm BinaryAlgorithm =
                 Training.TrainingAlgorithm.BINARY_STOCHASTIC_DUAL_COORDINATE_ASCENT;

        private const Training.TrainingAlgorithm MulticlassAlgorithm =
            Training.TrainingAlgorithm.STOCHASTIC_DUAL_COORDINATE_ASCENT;

        private const float LearningRate = 0.1f;
        
        private const  int MaxIterations = 1;

        private const int seed = 2;
#endregion
        
        
        public Program(string[] args)
        {
            var context = new MLContext(seed);
            var trainer = new Training(context);

            
            var containsPeopleModel = trainer.TrainModel(BinaryAlgorithm,
                new DataControl.TrainingOptions()
                {
                    FeatureColumn = "Features",
                    LabelColumn = "ContainsPeople",
                    LearningRate = LearningRate,
                    MaxIterations = MaxIterations
                });

            var numberOfPeopleModel = trainer.TrainModel(MulticlassAlgorithm,
                new DataControl.TrainingOptions()
                {
                    FeatureColumn = "Features",
                    LabelColumn = "NumberOfPeople",
                    LearningRate = LearningRate,
                    MaxIterations = MaxIterations
                }
            );
            
            var distanceModel = trainer.TrainModel(MulticlassAlgorithm,
                new DataControl.TrainingOptions()
                {
                    FeatureColumn = "Features",
                    LabelColumn = "Distance",
                    LearningRate = LearningRate,
                    MaxIterations = MaxIterations
                }
            );
            
            Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");

            trainer.Evaluate(containsPeopleModel,BinaryAlgorithm, "ContainsPeople");
            trainer.Evaluate(numberOfPeopleModel,MulticlassAlgorithm, "NumberOfPeople");
            trainer.Evaluate(distanceModel,MulticlassAlgorithm, "Distance");
           
            Console.WriteLine("=============== End of model evaluation ===============");
            
            Console.WriteLine("=============== Consuming Model ===============");
            Console.WriteLine();
            
            var predictor = new Predictor(context, containsPeopleModel, numberOfPeopleModel, distanceModel);

            List<DataControl.Data> data = new List<DataControl.Data>();
            foreach (var file in DataControl.FILES)
            {
                var tmp = DataControl.LoadTrainingData(file);
                data.AddRange(tmp.Skip(tmp.Count/2).Take(tmp.Count/2));
            }

            data = RandomPermutation(data);
            int incorrectContainsPeople = 0;
            int incorrectNumberOfPeople = 0;
            int incorrectDistance = 0;
            foreach (var d in data)
            {
                var result = predictor.Predict(d);
                if (result.ActualContainsPeople != result.PredictionContainsPeople)
                {
                    incorrectContainsPeople++;
                }
                if (result.ActualNumberOfPeople != result.PredictionNumberOfPeople)
                {
                    incorrectNumberOfPeople++;
                }
                if (result.ActualDistance != result.PredictionDistance)
                {
                    incorrectDistance++;
                }
            }

            Console.WriteLine();
            Console.WriteLine("Testing Results");
            Console.WriteLine("Incorrect Contains People: {0}/{1}  {2}%",incorrectContainsPeople,data.Count,(1 - ((double)incorrectContainsPeople/(double)data.Count))*100);
            Console.WriteLine("Incorrect Number of People: {0}/{1}  {2}%",incorrectNumberOfPeople,data.Count,(1 - ((double)incorrectNumberOfPeople/(double)data.Count))*100);
            Console.WriteLine("Incorrect Distance: {0}/{1}  {2}%",incorrectDistance,data.Count,(1 - ((double)incorrectDistance/(double)data.Count))*100);
            Console.WriteLine();
            Console.WriteLine("=========== Finished Consuming Model =============");
            
        }
        
        static Random random = new Random();
        private static List<T> RandomPermutation<T>(List<T> sequence)
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