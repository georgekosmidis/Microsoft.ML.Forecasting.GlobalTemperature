using Microsoft.ML.Forecasting.GlobalTemperature.Models;
using Microsoft.ML.Transforms.TimeSeries;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Engine
{

    public class Forecaster
    {
        /// <summary>
        /// The outcome of the forcase
        /// </summary>
        public ModelOutput Output { get; private set; }

        /// <summary>
        /// Forecasts a number of values
        /// </summary>
        /// <param name="context">The common context for all ML.NET operations.</param>
        /// <param name="trainer">A trainer to retrieve the transformer.</param>
        public Forecaster(MLContext context, Trainer trainer)
        {            
            var forecastEngine = trainer.Transformer.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);

            Output = forecastEngine.Predict();
        }


    }
}
