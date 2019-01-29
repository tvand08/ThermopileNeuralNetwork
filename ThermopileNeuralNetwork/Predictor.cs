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

        public PredictionResult Predict(Data data)
        {
            var predictedDistance = DistancePredictor.Predict(data);
            var predictedContainsPeople = ContainsPeoplePredictor.Predict(data);
            var predictedNumberOfPeople = NumberOfPeoplePredictor.Predict(data);

            return new PredictionResult()
            {
                ActualContainsPeople = data.ContainsPeople,
                ActualNumberOfPeople = data.NumberOfPeople,
                ActualDistance = data.Distance,
                PredictionContainsPeople = predictedContainsPeople.ContainsPeople,
                PredictionNumberOfPeople = predictedNumberOfPeople.NumberOfPeople - 1,
                PredictionDistance = predictedDistance.Distance - 1
            };
         
        }
    }
}