using Microsoft.ML.Forecasting.GlobalTemperature.Engine;
using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using Microsoft.ML.Forecasting.GlobalTemperature.UI;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Samples
{
    public class DemoPrediction : AbstractPrediction
    {
        int horizon = 12;//i know it is 12, but if you don't believe me:  dataLoader.TestData.GetColumn<DateTime>(nameof(ModelInput.Date)).Count();
        DateTime segment = DateTime.Parse("2015-01-01");

        public DemoPrediction()
        {
            
        }

        public void Run()
        {
            base.Prepare(segment);

            var trainer = new Trainer(base.MLContext, base.DataLoader, horizon);
            var evaluator = new Evaluator(base.MLContext, base.DataLoader, trainer);
            var forecaster = new Forecaster(base.MLContext, trainer);

            //UI
            var grid = new Grid(90);

            //General info
            grid.PrintLine();
            grid.PrintRow("Mean Absolute Error", evaluator.MAE.ToString("F3"));
            grid.PrintRow("Root Mean Squared Error", evaluator.RMSE.ToString("F3"));
            grid.PrintLine();

            //Evaluation results
            var testData = base.DataLoader.TestData;
            var predictions = forecaster.Predict();

            var forecastOutput =
                 base.MLContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                        .Take(horizon)
                        .Select((ModelInput model, int index) =>
                        {
                            var date = model.Date;
                            var actualTemps = model.LandAverageTemperature;
                            var lowerEstimate = predictions.LowerBoundLandAverageTemperature[index];
                            var estimate = predictions.ForecastedLandAverageTemperature[index];
                            var upperEstimate = predictions.UpperBoundLandAverageTemperature[index];
                            return new List<string> {
                                date.ToShortDateString(),
                                "Yellow|"+actualTemps.ToString("F3"),
                                "Yellow|"+estimate.ToString("F3"),
                                lowerEstimate.ToString("F3"),
                                upperEstimate.ToString("F3")
                            };
                        });

            grid.PrintLine();
            grid.PrintRow("Date", "Yellow|Actual", "Yellow|Forecase", "Lower", "Upper");
            grid.PrintLine();

            // Output predictions
            foreach (var prediction in forecastOutput)
            {
                grid.PrintRow(prediction.ToArray());
                grid.PrintLine();
            }
        }
    }
}
