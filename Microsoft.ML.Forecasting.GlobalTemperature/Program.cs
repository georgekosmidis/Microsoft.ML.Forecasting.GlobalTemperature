using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Forecasting.GlobalTemperature.Engine;
using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using Microsoft.ML.Forecasting.GlobalTemperature.UI;

namespace Microsoft.ML.Forecasting.GlobalTemperature
{
    class Program
    {
        static void Main(string[] args)
        {
            var rootDir = AppDomain.CurrentDomain.BaseDirectory;

            //Data from: https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data/data?select=GlobalTemperatures.csv
            var csvFilePath = Path.Combine(rootDir, "Data", "GlobalTemperatures.csv");
            
            var mlContext = new MLContext();
            var dataLoader = new DataLoader(
                    mlContext,
                    csvFilePath,
                    DateTime.Parse("2015-01-01")
            );
            var trainer = new Trainer(mlContext, dataLoader);
            var evaluator = new Evaluator(mlContext, dataLoader, trainer);
            var forecaster = new Forecaster(mlContext, trainer);

            //UI
            var grid = new Grid(90);
            
            //General info
            grid.PrintLine();
            grid.PrintRow("Mean Absolute Error", evaluator.MAE.ToString("F3"));
            grid.PrintRow("Root Mean Squared Error", evaluator.RMSE.ToString("F3"));
            grid.PrintLine();

            //Evaluation results
            var testData = dataLoader.TestData;

            var forecastOutput =
                 mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                        .Take(12)//i know it is 12, but if i didn't:  dataLoader.TestData.GetColumn<DateTime>(nameof(ModelInput.Date)).Count();
                        .Select((ModelInput rental, int index) =>
                        {
                            var date = rental.Date;
                            var actualTemps = rental.LandAverageTemperature;
                            var lowerEstimate = forecaster.Output.LowerBoundLandAverageTemperature[index];
                            var estimate = forecaster.Output.ForecastedLandAverageTemperature[index];
                            var upperEstimate = forecaster.Output.UpperBoundLandAverageTemperature[index];
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