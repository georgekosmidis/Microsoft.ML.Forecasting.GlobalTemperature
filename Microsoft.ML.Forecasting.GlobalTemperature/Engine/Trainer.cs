using System;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Engine
{
    public class Trainer
    {
        /// <summary>
        /// An <see cref="SsaForecastingTransformer"/> transformer
        /// </summary>
        public ITransformer Transformer { get; private set; }

        /// <summary>
        /// Trains a transformer. 
        /// </summary>
        /// <param name="context">The common context for all ML.NET operations.</param>
        /// <param name="dataLoader">A data loader instane to retrieve the train set.</param>
        public Trainer(MLContext context, DataLoader dataLoader)
        {          
            var seriesLength = dataLoader.SeriesData.GetColumn<DateTime>(nameof(ModelInput.Date)).Count();
            var trainSize = dataLoader.TrainData.GetColumn<DateTime>(nameof(ModelInput.Date)).Count();
            var testSize = dataLoader.TestData.GetColumn<DateTime>(nameof(ModelInput.Date)).Count();

            var forecastingPipeline = context.Forecasting.ForecastBySsa(
                outputColumnName: nameof(ModelOutput.ForecastedLandAverageTemperature),
                inputColumnName: nameof(ModelInput.LandAverageTemperature),
                windowSize: testSize,
                seriesLength: seriesLength,
                trainSize: trainSize,
                horizon: testSize,
                confidenceLevel: 0.95f,
                confidenceLowerBoundColumn: nameof(ModelOutput.LowerBoundLandAverageTemperature),
                confidenceUpperBoundColumn: nameof(ModelOutput.UpperBoundLandAverageTemperature));

            Transformer =  forecastingPipeline.Fit(dataLoader.TrainData);
        }

    }
}
