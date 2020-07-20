namespace Microsoft.ML.Forecasting.GlobalTemperature.Models
{
    public class ModelOutput
    {
        public float[] ForecastedLandAverageTemperature { get; set; }

        public float[] LowerBoundLandAverageTemperature { get; set; }

        public float[] UpperBoundLandAverageTemperature { get; set; }
    }
}
