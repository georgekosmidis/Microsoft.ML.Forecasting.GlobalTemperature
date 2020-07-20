using Microsoft.ML.Data;
using System;

namespace Microsoft.ML.Forecasting.GlobalTemperature.Models
{
    public class ModelInput
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float LandAverageTemperature { get; set; }

        [LoadColumn(2)]
        public float LandAverageTemperatureUncertainty { get; set; }

        [LoadColumn(3)]
        public float LandMaxTemperature { get; set; }

        [LoadColumn(4)]
        public float LandMaxTemperatureUncertainty { get; set; }

        [LoadColumn(5)]
        public float LandMinTemperature { get; set; }

        [LoadColumn(6)]
        public float LandMinTemperatureUncertainty { get; set; }

        [LoadColumn(7)]
        public float LandAndOceanAverageTemperature { get; set; }

        [LoadColumn(8)]
        public float LandAndOceanAverageTemperatureUncertainty { get; set; }
    }
}
