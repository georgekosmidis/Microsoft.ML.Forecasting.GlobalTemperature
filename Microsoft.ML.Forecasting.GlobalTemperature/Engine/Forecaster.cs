using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using Microsoft.ML.Transforms.TimeSeries;
using System;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Engine
{

    public class Forecaster
    {
        private TimeSeriesPredictionEngine<ModelInput, ModelOutput> timeSeriesPredictionEngine;

        /// <summary>
        /// Forecasts a number of values
        /// </summary>
        /// <param name="context">The common context for all ML.NET operations.</param>
        /// <param name="trainer">A trainer to retrieve the transformer.</param>
        public Forecaster(MLContext context, Trainer trainer)
        {
            timeSeriesPredictionEngine = trainer.Transformer.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);
        }

        public ModelOutput Predict()
        {
           return timeSeriesPredictionEngine.Predict();
        }

        public ModelOutput Predict(int futureValues)
        {
            return timeSeriesPredictionEngine.Predict(futureValues);
        }
    }
}
