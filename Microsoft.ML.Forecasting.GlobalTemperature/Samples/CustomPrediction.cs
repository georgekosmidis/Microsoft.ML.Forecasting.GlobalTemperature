using Microsoft.ML.Forecasting.GlobalTemperature.Engine;
using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using Microsoft.ML.Forecasting.GlobalTemperature.UI;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Samples
{
    public class CustomPrediction : AbstractPrediction
    {
        DateTime segment = DateTime.Parse("2015-01-01");

        public CustomPrediction()
        {
            base.Prepare(segment);
        }

        public CustomPrediction Header()
        {
            var grid = new Grid(90);
            grid.PrintLine();
            grid.PrintRow("Date", "Yellow|Forecase", "Lower", "Upper");
            grid.PrintLine();

            return this;
        }
        public void Run(int year, int month)
        {            
            var predictionForIndex = ((year - segment.Year) * 12) + month;

            var trainer = new Trainer(base.MLContext, base.DataLoader, predictionForIndex);
            var forecaster = new Forecaster(base.MLContext, trainer);

            //UI
            var grid = new Grid(90);

            //Results
            var predictions = forecaster.Predict(predictionForIndex);

            grid.PrintLine();
            //grid.PrintRow("Date", "Yellow|Forecase", "Lower", "Upper");
            //grid.PrintLine();
            grid.PrintRow(
                 DateTime.Parse(year + "-" + month + "-1").ToShortDateString(),
                 "Yellow|" + predictions.ForecastedLandAverageTemperature[predictionForIndex - 1].ToString("F3"),
                 predictions.LowerBoundLandAverageTemperature[predictionForIndex - 1].ToString("F3"),
                 predictions.UpperBoundLandAverageTemperature[predictionForIndex - 1].ToString("F3")
            );

        }
    }
}
