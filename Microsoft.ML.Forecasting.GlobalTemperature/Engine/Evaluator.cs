using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using System;
using System.Linq;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Engine
{
    public class Evaluator
    {
        /// <summary>
        /// The Mean Absolute Error
        /// </summary>
        public float MAE { get; private set; }

        /// <summary>
        /// The Root Mean Squared Error
        /// </summary>
        public double RMSE { get; private set; }

        /// <summary>
        /// Evaluates the model
        /// </summary>
        /// <param name="context">The common context for all ML.NET operations.</param>
        /// <param name="dataLoader">A DataLoader instance to retrieve the test data</param>
        /// <param name="trainer">A Trainer instance to retrieve an ITransformer</param>
        public Evaluator(MLContext context, DataLoader dataLoader, Trainer trainer)
        {
            var model = trainer.Transformer;
            var testData = dataLoader.TestData;

            // Make predictions
            var predictions = model.Transform(testData);

            // Actual values
            var actual =
                context.Data.CreateEnumerable<ModelInput>(testData, true)
                    .Select(observed => observed.LandAverageTemperature);

            // Predicted values
            var forecast =
                context.Data.CreateEnumerable<ModelOutput>(predictions, true)
                    .Select(prediction => prediction.ForecastedLandAverageTemperature[0]);

            // Calculate error (actual - forecast)
            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            // Get metric averages
            MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error
        }
    }
}
