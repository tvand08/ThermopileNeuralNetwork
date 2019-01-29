using System;
using System.Data;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using static ThermopileNeuralNetwork.DataControl;
namespace ThermopileNeuralNetwork
{
    public class Predictor
    {
        
        private PredictionEngine<Data,PredictedContainsPeople> ContainsPeoplePredictor { get; set; }
        private PredictionEngine<Data,PredictedNumberOfPeople> NumberOfPeoplePredictor { get; set; }
        private PredictionEngine<Data,PredictedDistance> DistancePredictor { get; set; }

        public Predictor(MLContext context,ITransformer containsPeopleModel, ITransformer numberOfPeopleModel, ITransformer distanceModel)
        {
            ContainsPeoplePredictor = containsPeopleModel.CreatePredictionEngine<Data, PredictedContainsPeople>(context);
            NumberOfPeoplePredictor = numberOfPeopleModel.CreatePredictionEngine<Data, PredictedNumberOfPeople>(context);
            DistancePredictor = distanceModel.CreatePredictionEngine<Data, PredictedDistance>(context);
            
        }

        public void Predict(Data data)
        {
            var predictedDistance = DistancePredictor.Predict(data);
            var predictedContainsPeople = ContainsPeoplePredictor.Predict(data);
            var predictedNumberOfPeople = NumberOfPeoplePredictor.Predict(data);
            
            Console.WriteLine("Actual Contains People: {0}  Predicted Contains People: {1}",data.ContainsPeople,predictedContainsPeople.ContainsPeople);
            Console.WriteLine("Actual Number Of People: {0}  Predicted Number of People: {1}",data.NumberOfPeople,predictedNumberOfPeople.NumberOfPeople);
            Console.WriteLine("Actual Distance: {0}  Predicted Distance: {1}",data.Distance,predictedDistance.Distance);
            Console.WriteLine();
        }
    }
}